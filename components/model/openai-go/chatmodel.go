/*
 * Copyright 2026 CloudWeGo Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package openaigo

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"runtime/debug"
	"strings"
	"time"

	"github.com/cloudwego/eino/callbacks"
	"github.com/cloudwego/eino/components"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

var _ model.ToolCallingChatModel = (*ChatModel)(nil)

type Config struct {
	APIKey string `json:"api_key"`

	// Timeout specifies the maximum duration to wait for API responses.
	// If HTTPClient is set, Timeout will not be used.
	// Optional. Default: no timeout
	Timeout time.Duration `json:"timeout"`

	// HTTPClient specifies the client to send HTTP requests.
	// If HTTPClient is set, Timeout will not be used.
	// Optional. Default &http.Client{Timeout: Timeout}
	HTTPClient *http.Client `json:"http_client"`

	// BaseURL specifies the OpenAI endpoint URL
	// Optional. Default: https://api.openai.com/v1
	BaseURL string `json:"base_url"`

	// Model specifies the ID of the model to use.
	// Optional.
	Model string `json:"model,omitempty"`

	// MaxOutputTokens is an upper bound for the number of tokens that can be generated for a response,
	// including visible output tokens and reasoning tokens.
	MaxOutputTokens *int `json:"max_output_tokens,omitempty"`

	TopP        *float32 `json:"top_p,omitempty"`
	Temperature *float32 `json:"temperature,omitempty"`

	// Reasoning config for reasoning models.
	Reasoning *Reasoning `json:"reasoning,omitempty"`

	// Store indicates whether to store the generated model response for later retrieval.
	Store *bool `json:"store,omitempty"`

	// Metadata set of key-value pairs that can be attached to an object.
	Metadata map[string]string `json:"metadata,omitempty"`

	// ExtraFields will override any existing fields with the same key.
	// Optional. Useful for experimental features not yet officially supported.
	ExtraFields map[string]any `json:"extra_fields,omitempty"`
}

type ChatModel struct {
	cli openai.Client

	model       string
	maxOutTok   *int
	topP        *float32
	temperature *float32
	reasoning   *Reasoning
	store       *bool
	metadata    map[string]string
	extraFields map[string]any

	tools      []responses.ToolUnionParam
	rawTools   []*schema.ToolInfo
	toolChoice *schema.ToolChoice
}

func NewChatModel(_ context.Context, config *Config) (*ChatModel, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}

	opts := make([]option.RequestOption, 0, 4)
	if config.APIKey != "" {
		opts = append(opts, option.WithAPIKey(config.APIKey))
	}
	if config.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(config.BaseURL))
	}
	if config.HTTPClient != nil {
		opts = append(opts, option.WithHTTPClient(config.HTTPClient))
	} else if config.Timeout > 0 {
		opts = append(opts, option.WithHTTPClient(&http.Client{Timeout: config.Timeout}))
	}

	cli := openai.NewClient(opts...)

	cm := &ChatModel{
		cli:         cli,
		model:       config.Model,
		maxOutTok:   config.MaxOutputTokens,
		topP:        config.TopP,
		temperature: config.Temperature,
		reasoning:   config.Reasoning,
		store:       config.Store,
		metadata:    cloneStringMap(config.Metadata),
		extraFields: cloneAnyMap(config.ExtraFields),
	}

	return cm, nil
}

func (cm *ChatModel) Generate(ctx context.Context, in []*schema.Message, opts ...model.Option) (outMsg *schema.Message, err error) {
	ctx = callbacks.EnsureRunInfo(ctx, cm.GetType(), components.ComponentOfChatModel)

	params, cbIn, err := cm.buildParams(in, false, opts...)
	if err != nil {
		return nil, err
	}

	ctx = callbacks.OnStart(ctx, cbIn)
	defer func() {
		if err != nil {
			callbacks.OnError(ctx, err)
		}
	}()

	resp, err := cm.cli.Responses.New(ctx, params)
	if err != nil {
		return nil, err
	}

	outMsg, err = cm.convertResponseToMessage(resp)
	if err != nil {
		return nil, err
	}

	callbacks.OnEnd(ctx, &model.CallbackOutput{
		Message:    outMsg,
		Config:     cbIn.Config,
		TokenUsage: toModelTokenUsage(outMsg.ResponseMeta),
		Extra: map[string]any{
			callbackExtraModelName: string(resp.Model),
		},
	})

	return outMsg, nil
}

func (cm *ChatModel) Stream(ctx context.Context, in []*schema.Message, opts ...model.Option) (outStream *schema.StreamReader[*schema.Message], err error) {
	ctx = callbacks.EnsureRunInfo(ctx, cm.GetType(), components.ComponentOfChatModel)

	params, cbIn, err := cm.buildParams(in, true, opts...)
	if err != nil {
		return nil, err
	}

	ctx = callbacks.OnStart(ctx, cbIn)
	defer func() {
		if err != nil {
			callbacks.OnError(ctx, err)
		}
	}()

	stream := cm.cli.Responses.NewStreaming(ctx, params)

	sr, sw := schema.Pipe[*model.CallbackOutput](1)
	go func() {
		defer func() {
			pe := recover()
			_ = stream.Close()
			if pe != nil {
				_ = sw.Send(nil, newPanicErr(pe, debug.Stack()))
			}
			sw.Close()
		}()

		state := newStreamState()
		for stream.Next() {
			ev := stream.Current()
			msg, done, deltaOnly, err2 := state.consume(ev)
			if err2 != nil {
				_ = sw.Send(nil, err2)
				return
			}
			if msg == nil {
				continue
			}

			// ensure callbacks can receive token usage on final chunk.
			if !deltaOnly {
				msg.ResponseMeta = ensureResponseMeta(msg.ResponseMeta)
			}

			closed := sw.Send(&model.CallbackOutput{
				Message:    msg,
				Config:     cbIn.Config,
				TokenUsage: toModelTokenUsage(msg.ResponseMeta),
				Extra: func() map[string]any {
					if done && state.modelName != "" {
						return map[string]any{callbackExtraModelName: state.modelName}
					}
					return nil
				}(),
			}, nil)
			if closed {
				return
			}
		}

		if stream.Err() != nil {
			_ = sw.Send(nil, stream.Err())
			return
		}
	}()

	ctx, nsr := callbacks.OnEndWithStreamOutput(ctx, schema.StreamReaderWithConvert(sr,
		func(src *model.CallbackOutput) (callbacks.CallbackOutput, error) { return src, nil },
	))

	outStream = schema.StreamReaderWithConvert(nsr, func(src callbacks.CallbackOutput) (*schema.Message, error) {
		s := src.(*model.CallbackOutput)
		if s.Message == nil {
			return nil, schema.ErrNoValue
		}
		return s.Message, nil
	})

	return outStream, nil
}

func (cm *ChatModel) WithTools(tools []*schema.ToolInfo) (model.ToolCallingChatModel, error) {
	if len(tools) == 0 {
		return nil, errors.New("no tools to bind")
	}
	openAITools, rawTools, err := toOpenAITools(tools)
	if err != nil {
		return nil, err
	}

	tc := schema.ToolChoiceAllowed
	ncm := *cm
	ncm.tools = openAITools
	ncm.rawTools = rawTools
	ncm.toolChoice = &tc
	return &ncm, nil
}

const typ = "OpenAI"

func (cm *ChatModel) GetType() string { return typ }

func (cm *ChatModel) IsCallbacksEnabled() bool { return true }

// ------------------------ params ------------------------

func (cm *ChatModel) buildParams(in []*schema.Message, stream bool, opts ...model.Option) (responses.ResponseNewParams, *model.CallbackInput, error) {
	common := model.GetCommonOptions(&model.Options{
		Temperature: cm.temperature,
		MaxTokens: func() *int {
			// Responses API uses MaxOutputTokens; keep MaxTokens in common opts unused.
			return nil
		}(),
		Model:      &cm.model,
		TopP:       cm.topP,
		Tools:      cm.rawTools,
		ToolChoice: cm.toolChoice,
	}, opts...)

	spec := model.GetImplSpecificOptions(&options{
		MaxOutputTokens: cm.maxOutTok,
		Reasoning:       cm.reasoning,
		Store:           cm.store,
		Metadata:        cm.metadata,
		ExtraFields:     cm.extraFields,
	}, opts...)

	params := responses.ResponseNewParams{}
	if common.Model != nil {
		params.Model = responsesModelFromString(*common.Model)
	}
	if spec.MaxOutputTokens != nil {
		params.MaxOutputTokens = openai.Int(int64(*spec.MaxOutputTokens))
	}
	if common.Temperature != nil {
		params.Temperature = openai.Float(float64(*common.Temperature))
	}
	if common.TopP != nil {
		params.TopP = openai.Float(float64(*common.TopP))
	}
	if spec.Store != nil {
		params.Store = openai.Bool(*spec.Store)
	}
	if len(spec.Metadata) > 0 {
		params.Metadata = spec.Metadata
	}
	if spec.Reasoning != nil {
		params.Reasoning = spec.Reasoning.toSDK()
	}
	if stream {
		params.StreamOptions = responses.ResponseNewParamsStreamOptions{IncludeObfuscation: openai.Bool(false)}
	}

	// Tools.
	tools := cm.tools
	cbTools := cm.rawTools
	if common.Tools != nil {
		var err error
		tools, cbTools, err = toOpenAITools(common.Tools)
		if err != nil {
			return responses.ResponseNewParams{}, nil, err
		}
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	if err := populateToolChoice(&params, common.ToolChoice, common.AllowedToolNames, tools); err != nil {
		return responses.ResponseNewParams{}, nil, err
	}

	// Input.
	inputItems, err := toInputItems(in)
	if err != nil {
		return responses.ResponseNewParams{}, nil, err
	}
	params.Input = responses.ResponseNewParamsInputUnion{OfInputItemList: inputItems}

	if len(spec.ExtraFields) > 0 {
		params.SetExtraFields(spec.ExtraFields)
	}

	cbIn := &model.CallbackInput{
		Messages:   in,
		Tools:      cbTools,
		ToolChoice: common.ToolChoice,
		Config: &model.Config{
			Model:       string(params.Model),
			MaxTokens:   int(optInt64(params.MaxOutputTokens)),
			Temperature: float32(optFloat64(params.Temperature)),
			TopP:        float32(optFloat64(params.TopP)),
		},
	}

	return params, cbIn, nil
}

func toInputItems(in []*schema.Message) (responses.ResponseInputParam, error) {
	items := make([]responses.ResponseInputItemUnionParam, 0, len(in))
	for _, msg := range in {
		if msg == nil {
			continue
		}
		switch msg.Role {
		case schema.User:
			content, err := toInputContentFromMessage(msg)
			if err != nil {
				return nil, err
			}
			items = append(items, responses.ResponseInputItemParamOfMessage(content, responses.EasyInputMessageRoleUser))
		case schema.System:
			content, err := toInputContentFromMessage(msg)
			if err != nil {
				return nil, err
			}
			items = append(items, responses.ResponseInputItemParamOfMessage(content, responses.EasyInputMessageRoleSystem))
		case schema.Assistant:
			// Assistant messages are previous model outputs. The Responses API is strict:
			// when role=assistant, content parts must be of type "output_text"/"refusal",
			// not "input_text".
			//
			// The openai-go SDK's easiest compatible representation is to send assistant
			// content as a plain string (not a typed content-part list).
			// We therefore:
			//   - allow text-only assistant history (as string)
			//   - reject non-text assistant multimodal content when re-sending history
			assistantText, hasAssistantText, err := extractAssistantTextForHistory(msg)
			if err != nil {
				return nil, err
			}
			if hasAssistantText {
				items = append(items, responses.ResponseInputItemParamOfMessage(assistantText, responses.EasyInputMessageRoleAssistant))
			}

			// assistant tool calls
			for _, tc := range msg.ToolCalls {
				items = append(items, responses.ResponseInputItemParamOfFunctionCall(tc.Function.Arguments, tc.ID, tc.Function.Name))
			}
		case schema.Tool:
			// tool call output
			if msg.ToolCallID == "" {
				return nil, fmt.Errorf("tool message missing ToolCallID")
			}
			if len(msg.UserInputMultiContent) == 0 {
				items = append(items, responses.ResponseInputItemParamOfFunctionCallOutput(msg.ToolCallID, msg.Content))
				break
			}
			outItems := make([]responses.ResponseFunctionCallOutputItemUnionParam, 0, len(msg.UserInputMultiContent))
			for _, part := range msg.UserInputMultiContent {
				switch part.Type {
				case schema.ChatMessagePartTypeText:
					outItems = append(outItems, responses.ResponseFunctionCallOutputItemUnionParam{OfInputText: &responses.ResponseInputTextContentParam{Text: part.Text}})
				case schema.ChatMessagePartTypeImageURL:
					if part.Image == nil {
						return nil, fmt.Errorf("image field must not be nil in tool message")
					}
					url, err := commonToDataOrURL(part.Image.MessagePartCommon)
					if err != nil {
						return nil, err
					}
					outItems = append(outItems, responses.ResponseFunctionCallOutputItemUnionParam{OfInputImage: &responses.ResponseInputImageContentParam{ImageURL: openai.String(url)}})
				case schema.ChatMessagePartTypeFileURL:
					if part.File == nil {
						return nil, fmt.Errorf("file field must not be nil in tool message")
					}
					url, err := commonToDataOrURL(part.File.MessagePartCommon)
					if err != nil {
						return nil, err
					}
					p := &responses.ResponseInputFileContentParam{}
					if part.File.URL != nil {
						p.FileURL = openai.String(url)
					} else {
						p.FileData = openai.String(url)
					}
					if part.File.Name != "" {
						p.Filename = openai.String(part.File.Name)
					}
					outItems = append(outItems, responses.ResponseFunctionCallOutputItemUnionParam{OfInputFile: p})
				default:
					return nil, fmt.Errorf("unsupported tool output content type: %s", part.Type)
				}
			}
			items = append(items, responses.ResponseInputItemParamOfFunctionCallOutput(msg.ToolCallID, responses.ResponseFunctionCallOutputItemListParam(outItems)))
		default:
			return nil, fmt.Errorf("unknown role: %s", msg.Role)
		}
	}

	return items, nil
}

func extractAssistantTextForHistory(msg *schema.Message) (text string, ok bool, err error) {
	if msg == nil {
		return "", false, nil
	}

	// Prefer the canonical Content field.
	if msg.Content != "" {
		return msg.Content, true, nil
	}

	// If Content is empty, attempt to derive text from multi-content.
	// If any non-text part exists, we fail fast to avoid producing invalid request bodies.
	if len(msg.AssistantGenMultiContent) > 0 {
		var b strings.Builder
		for _, part := range msg.AssistantGenMultiContent {
			if part.Type != schema.ChatMessagePartTypeText {
				return "", false, fmt.Errorf("assistant history contains non-text part (%s); cannot re-send as Responses API input", part.Type)
			}
			if part.Text == "" {
				continue
			}
			if b.Len() > 0 {
				b.WriteString("\n")
			}
			b.WriteString(part.Text)
		}
		if b.Len() > 0 {
			return b.String(), true, nil
		}
	}

	// Deprecated MultiContent.
	if len(msg.MultiContent) > 0 {
		var b strings.Builder
		for _, c := range msg.MultiContent {
			if c.Type != schema.ChatMessagePartTypeText {
				return "", false, fmt.Errorf("assistant history contains deprecated MultiContent non-text part (%s); cannot re-send", c.Type)
			}
			if c.Text == "" {
				continue
			}
			if b.Len() > 0 {
				b.WriteString("\n")
			}
			b.WriteString(c.Text)
		}
		if b.Len() > 0 {
			return b.String(), true, nil
		}
	}

	// Do not attempt to re-send UserInputMultiContent on assistant messages.
	if len(msg.UserInputMultiContent) > 0 {
		return "", false, fmt.Errorf("assistant history contains UserInputMultiContent; cannot re-send as Responses API input")
	}

	return "", false, nil
}

func toInputContentFromMessage(msg *schema.Message) (responses.ResponseInputMessageContentListParam, error) {
	if len(msg.UserInputMultiContent) > 0 && len(msg.AssistantGenMultiContent) > 0 {
		return nil, fmt.Errorf("a message cannot contain both UserInputMultiContent and AssistantGenMultiContent")
	}
	if len(msg.UserInputMultiContent) > 0 {
		parts := make([]responses.ResponseInputContentUnionParam, 0, len(msg.UserInputMultiContent))
		for _, part := range msg.UserInputMultiContent {
			p, err := toInputContentPartFromInputPart(part)
			if err != nil {
				return nil, err
			}
			parts = append(parts, p)
		}
		return responses.ResponseInputMessageContentListParam(parts), nil
	}
	if len(msg.AssistantGenMultiContent) > 0 {
		// For assistant messages, only text parts can be re-sent as input.
		parts := make([]responses.ResponseInputContentUnionParam, 0, len(msg.AssistantGenMultiContent))
		for _, part := range msg.AssistantGenMultiContent {
			if part.Type != schema.ChatMessagePartTypeText {
				return nil, fmt.Errorf("unsupported assistant output part type in re-input: %s", part.Type)
			}
			parts = append(parts, responses.ResponseInputContentUnionParam{OfInputText: &responses.ResponseInputTextParam{Text: part.Text}})
		}
		return responses.ResponseInputMessageContentListParam(parts), nil
	}

	// Backward compatible deprecated MultiContent.
	if len(msg.MultiContent) > 0 {
		parts := make([]responses.ResponseInputContentUnionParam, 0, len(msg.MultiContent))
		for _, c := range msg.MultiContent {
			switch c.Type {
			case schema.ChatMessagePartTypeText:
				parts = append(parts, responses.ResponseInputContentUnionParam{OfInputText: &responses.ResponseInputTextParam{Text: c.Text}})
			case schema.ChatMessagePartTypeImageURL:
				if c.ImageURL == nil {
					continue
				}
				parts = append(parts, responses.ResponseInputContentUnionParam{OfInputImage: &responses.ResponseInputImageParam{
					Detail: responses.ResponseInputImageDetailAuto,
					ImageURL: openai.String(func() string {
						if c.ImageURL.URI != "" {
							return c.ImageURL.URI
						}
						return c.ImageURL.URL
					}()),
				}})
			default:
				return nil, fmt.Errorf("unsupported deprecated MultiContent part type: %s", c.Type)
			}
		}
		return responses.ResponseInputMessageContentListParam(parts), nil
	}

	if msg.Content == "" {
		// allow empty content for assistant messages
		if msg.Role == schema.Assistant {
			return responses.ResponseInputMessageContentListParam([]responses.ResponseInputContentUnionParam{}), nil
		}
		return nil, fmt.Errorf("message content is empty")
	}
	return responses.ResponseInputMessageContentListParam([]responses.ResponseInputContentUnionParam{{
		OfInputText: &responses.ResponseInputTextParam{Text: msg.Content},
	}}), nil
}

func toInputContentPartFromInputPart(part schema.MessageInputPart) (responses.ResponseInputContentUnionParam, error) {
	switch part.Type {
	case schema.ChatMessagePartTypeText:
		return responses.ResponseInputContentUnionParam{OfInputText: &responses.ResponseInputTextParam{Text: part.Text}}, nil
	case schema.ChatMessagePartTypeImageURL:
		if part.Image == nil {
			return responses.ResponseInputContentUnionParam{}, fmt.Errorf("image field must not be nil when type is %s", part.Type)
		}
		url, err := commonToDataOrURL(part.Image.MessagePartCommon)
		if err != nil {
			return responses.ResponseInputContentUnionParam{}, err
		}
		return responses.ResponseInputContentUnionParam{OfInputImage: &responses.ResponseInputImageParam{
			Detail:   toSDKImageDetail(part.Image.Detail),
			ImageURL: openai.String(url),
		}}, nil
	case schema.ChatMessagePartTypeFileURL:
		if part.File == nil {
			return responses.ResponseInputContentUnionParam{}, fmt.Errorf("file field must not be nil when type is %s", part.Type)
		}
		fileURL, err := commonToDataOrURL(part.File.MessagePartCommon)
		if err != nil {
			return responses.ResponseInputContentUnionParam{}, err
		}
		fileParam := &responses.ResponseInputFileParam{}
		if part.File.URL != nil {
			fileParam.FileURL = openai.String(fileURL)
		} else if part.File.Base64Data != nil {
			fileParam.FileData = openai.String(fileURL)
		}
		if part.File.Name != "" {
			fileParam.Filename = openai.String(part.File.Name)
		}
		return responses.ResponseInputContentUnionParam{OfInputFile: fileParam}, nil
	default:
		return responses.ResponseInputContentUnionParam{}, fmt.Errorf("unsupported content type: %s", part.Type)
	}
}

// Deprecated: tool call outputs are constructed inline in toInputItems.
func toFunctionCallOutputFromToolMessage(_ *schema.Message) (any, error) {
	return nil, fmt.Errorf("deprecated")
}

func populateToolChoice(params *responses.ResponseNewParams, tc *schema.ToolChoice, allowedToolNames []string, tools []responses.ToolUnionParam) error {
	if tc == nil {
		return nil
	}

	switch *tc {
	case schema.ToolChoiceForbidden:
		params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: openai.Opt(responses.ToolChoiceOptionsNone)}
		return nil
	case schema.ToolChoiceAllowed:
		params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: openai.Opt(responses.ToolChoiceOptionsAuto)}
		return nil
	case schema.ToolChoiceForced:
		if len(tools) == 0 {
			return fmt.Errorf("tool_choice is forced but no tools are provided")
		}

		// If a single allowed tool is specified (or only one tool exists), force it.
		var onlyOneToolName string
		if len(allowedToolNames) > 0 {
			if len(allowedToolNames) > 1 {
				return fmt.Errorf("only one allowed tool name can be configured")
			}
			allowed := allowedToolNames[0]
			if !toolNameExists(tools, allowed) {
				return fmt.Errorf("allowed tool name '%s' not found in tools list", allowed)
			}
			onlyOneToolName = allowed
		} else if len(tools) == 1 {
			if tools[0].OfFunction != nil {
				onlyOneToolName = tools[0].OfFunction.Name
			}
		}

		if onlyOneToolName != "" {
			params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{OfFunctionTool: &responses.ToolChoiceFunctionParam{Name: onlyOneToolName}}
			return nil
		}

		params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: openai.Opt(responses.ToolChoiceOptionsRequired)}
		return nil
	default:
		return fmt.Errorf("unknown tool choice: %s", *tc)
	}
}

func toolNameExists(tools []responses.ToolUnionParam, name string) bool {
	for _, t := range tools {
		if t.OfFunction != nil && t.OfFunction.Name == name {
			return true
		}
	}
	return false
}

func toOpenAITools(tis []*schema.ToolInfo) ([]responses.ToolUnionParam, []*schema.ToolInfo, error) {
	tools := make([]responses.ToolUnionParam, len(tis))
	rawTools := make([]*schema.ToolInfo, len(tis))
	copy(rawTools, tis)
	for i := range tis {
		ti := tis[i]
		if ti == nil {
			return nil, nil, fmt.Errorf("tool info cannot be nil")
		}
		paramsJSONSchema, err := ti.ParamsOneOf.ToJSONSchema()
		if err != nil {
			return nil, nil, fmt.Errorf("failed to convert tool parameters to JSONSchema: %w", err)
		}
		paramsMap, err := jsonSchemaToMap(paramsJSONSchema)
		if err != nil {
			return nil, nil, err
		}
		// The OpenAI Responses API requires strict tool schemas to include:
		//   - type: "object"
		//   - properties: {...}
		//   - additionalProperties: false
		//   - required: [all keys in properties]
		// Many JSON Schema generators omit "required" for fields tagged with `omitempty`.
		enforceOpenAIStrictJSONSchema(paramsMap)
		t := responses.ToolUnionParam{OfFunction: &responses.FunctionToolParam{
			Name:        ti.Name,
			Description: openai.String(ti.Desc),
			Parameters:  paramsMap,
			Strict:      openai.Bool(true),
		}}
		tools[i] = t
	}
	return tools, rawTools, nil
}

func enforceOpenAIStrictJSONSchema(schema map[string]any) {
	if schema == nil {
		return
	}

	// Recurse into nested schemas first.
	if items, ok := schema["items"]; ok {
		switch v := items.(type) {
		case map[string]any:
			enforceOpenAIStrictJSONSchema(v)
		case []any:
			for _, it := range v {
				if m, ok := it.(map[string]any); ok {
					enforceOpenAIStrictJSONSchema(m)
				}
			}
		}
	}
	if props, ok := schema["properties"].(map[string]any); ok {
		for _, pv := range props {
			if pm, ok := pv.(map[string]any); ok {
				enforceOpenAIStrictJSONSchema(pm)
			}
		}
	}
	if oneOf, ok := schema["oneOf"].([]any); ok {
		for _, ov := range oneOf {
			if om, ok := ov.(map[string]any); ok {
				enforceOpenAIStrictJSONSchema(om)
			}
		}
	}
	if anyOf, ok := schema["anyOf"].([]any); ok {
		for _, av := range anyOf {
			if am, ok := av.(map[string]any); ok {
				enforceOpenAIStrictJSONSchema(am)
			}
		}
	}
	if allOf, ok := schema["allOf"].([]any); ok {
		for _, av := range allOf {
			if am, ok := av.(map[string]any); ok {
				enforceOpenAIStrictJSONSchema(am)
			}
		}
	}

	// Now enforce strictness for object schemas.
	props, ok := schema["properties"].(map[string]any)
	if !ok || len(props) == 0 {
		return
	}

	// Ensure type is object (some generators omit it at the top level).
	if _, ok := schema["type"]; !ok {
		schema["type"] = "object"
	}

	// OpenAI strict schema expects additionalProperties=false.
	if _, ok := schema["additionalProperties"]; !ok {
		schema["additionalProperties"] = false
	}

	// Ensure required includes *all* keys in properties.
	existing := map[string]struct{}{}
	if req, ok := schema["required"]; ok {
		switch v := req.(type) {
		case []any:
			for _, it := range v {
				if s, ok := it.(string); ok {
					existing[s] = struct{}{}
				}
			}
		case []string:
			for _, s := range v {
				existing[s] = struct{}{}
			}
		}
	}

	required := make([]any, 0, len(props))
	for k := range props {
		if _, ok := existing[k]; !ok {
			existing[k] = struct{}{}
		}
	}
	for k := range existing {
		required = append(required, k)
	}

	// If there were no required keys produced for some reason, at least include all properties.
	if len(required) == 0 {
		for k := range props {
			required = append(required, k)
		}
	}

	schema["required"] = required
}

func jsonSchemaToMap(s any) (map[string]any, error) {
	if s == nil {
		return map[string]any{}, nil
	}
	// jsonschema.Schema has json tags; encode/decode to map[string]any.
	b, err := json.Marshal(s)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	return m, nil
}

func toSDKImageDetail(detail schema.ImageURLDetail) responses.ResponseInputImageDetail {
	switch detail {
	case schema.ImageURLDetailHigh:
		return responses.ResponseInputImageDetailHigh
	case schema.ImageURLDetailLow:
		return responses.ResponseInputImageDetailLow
	case schema.ImageURLDetailAuto:
		return responses.ResponseInputImageDetailAuto
	default:
		return responses.ResponseInputImageDetailAuto
	}
}

func commonToDataOrURL(common schema.MessagePartCommon) (string, error) {
	if common.URL == nil && common.Base64Data == nil {
		return "", fmt.Errorf("message part must have URL or Base64Data")
	}
	if common.URL != nil {
		return *common.URL, nil
	}
	if common.MIMEType == "" {
		return "", fmt.Errorf("message part must have MIMEType when using Base64Data")
	}
	if strings.HasPrefix(*common.Base64Data, "data:") {
		return "", fmt.Errorf("base64Data must be raw base64 without 'data:' prefix")
	}
	return fmt.Sprintf("data:%s;base64,%s", common.MIMEType, *common.Base64Data), nil
}

// ------------------------ response conversion ------------------------

func (cm *ChatModel) convertResponseToMessage(resp *responses.Response) (*schema.Message, error) {
	if resp == nil {
		return nil, fmt.Errorf("nil response")
	}

	msg := &schema.Message{Role: schema.Assistant}
	msg.ResponseMeta = ensureResponseMeta(msg.ResponseMeta)
	msg.ResponseMeta.FinishReason = string(resp.Status)
	msg.ResponseMeta.Usage = toEinoTokenUsage(resp.Usage)

	// Extract tool calls and assistant text.
	msg.Content = resp.OutputText()
	for _, item := range resp.Output {
		switch v := item.AsAny().(type) {
		case responses.ResponseFunctionToolCall:
			msg.ToolCalls = append(msg.ToolCalls, schema.ToolCall{
				ID:   v.CallID,
				Type: "function",
				Function: schema.FunctionCall{
					Name:      v.Name,
					Arguments: v.Arguments,
				},
			})
		case responses.ResponseOutputItemImageGenerationCall:
			// result is base64 image (no data: prefix)
			if v.Result != "" {
				b64 := v.Result
				msg.AssistantGenMultiContent = append(msg.AssistantGenMultiContent, schema.MessageOutputPart{
					Type: schema.ChatMessagePartTypeImageURL,
					Image: &schema.MessageOutputImage{
						MessagePartCommon: schema.MessagePartCommon{
							Base64Data: &b64,
							MIMEType:   "image/png",
						},
					},
				})
			}
		case responses.ResponseReasoningItem:
			// Prefer summary text when provided; otherwise content.
			msg.ReasoningContent = joinReasoningText(v)
		}
	}

	if len(msg.Content) > 0 {
		// keep assistant text as first part if no parts exist yet.
		if len(msg.AssistantGenMultiContent) == 0 {
			msg.AssistantGenMultiContent = append(msg.AssistantGenMultiContent, schema.MessageOutputPart{
				Type: schema.ChatMessagePartTypeText,
				Text: msg.Content,
			})
		} else {
			// prepend text part to existing parts
			msg.AssistantGenMultiContent = append([]schema.MessageOutputPart{{
				Type: schema.ChatMessagePartTypeText,
				Text: msg.Content,
			}}, msg.AssistantGenMultiContent...)
		}
	}

	return msg, nil
}

func joinReasoningText(item responses.ResponseReasoningItem) string {
	// Summary is often what people want.
	if len(item.Summary) > 0 {
		var b strings.Builder
		for i, s := range item.Summary {
			if s.Text == "" {
				continue
			}
			if i > 0 {
				b.WriteString("\n\n")
			}
			b.WriteString(s.Text)
		}
		out := b.String()
		if out != "" {
			return out
		}
	}

	if len(item.Content) > 0 {
		var b strings.Builder
		for i, c := range item.Content {
			if c.Text == "" {
				continue
			}
			if i > 0 {
				b.WriteString("\n\n")
			}
			b.WriteString(c.Text)
		}
		return b.String()
	}

	return ""
}

func toEinoTokenUsage(usage responses.ResponseUsage) *schema.TokenUsage {
	// usage is a value type; if it is all zeros, treat as absent.
	if usage.InputTokens == 0 && usage.OutputTokens == 0 && usage.TotalTokens == 0 {
		return nil
	}
	return &schema.TokenUsage{
		PromptTokens: int(usage.InputTokens),
		PromptTokenDetails: schema.PromptTokenDetails{
			CachedTokens: int(usage.InputTokensDetails.CachedTokens),
		},
		CompletionTokens: int(usage.OutputTokens),
		TotalTokens:      int(usage.TotalTokens),
		CompletionTokensDetails: schema.CompletionTokensDetails{
			ReasoningTokens: int(usage.OutputTokensDetails.ReasoningTokens),
		},
	}
}

func toModelTokenUsage(meta *schema.ResponseMeta) *model.TokenUsage {
	if meta == nil || meta.Usage == nil {
		return nil
	}
	u := meta.Usage
	return &model.TokenUsage{
		PromptTokens: u.PromptTokens,
		PromptTokenDetails: model.PromptTokenDetails{
			CachedTokens: u.PromptTokenDetails.CachedTokens,
		},
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
		CompletionTokensDetails: model.CompletionTokensDetails{
			ReasoningTokens: u.CompletionTokensDetails.ReasoningTokens,
		},
	}
}

func ensureResponseMeta(meta *schema.ResponseMeta) *schema.ResponseMeta {
	if meta == nil {
		return &schema.ResponseMeta{}
	}
	return meta
}

func responsesModelFromString(s string) responses.ResponsesModel { return shared.ResponsesModel(s) }

func optInt64(v param.Opt[int64]) int64 {
	if v.Valid() {
		return v.Value
	}
	return 0
}

func optFloat64(v param.Opt[float64]) float64 {
	if v.Valid() {
		return v.Value
	}
	return 0
}

// ------------------------ misc ------------------------

const callbackExtraModelName = "model_name"

type panicErr struct {
	info  any
	stack []byte
}

func (p *panicErr) Error() string {
	return fmt.Sprintf("panic error: %v, \nstack: %s", p.info, string(p.stack))
}

func newPanicErr(info any, stack []byte) error {
	return &panicErr{info: info, stack: stack}
}

func cloneStringMap(in map[string]string) map[string]string {
	if in == nil {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func cloneAnyMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

// consume and map streaming events into eino messages.
type streamState struct {
	modelName       string
	functionArgBufs map[string]*strings.Builder // key: item_id
	callIDByItemID  map[string]string
	nameByItemID    map[string]string
}

func newStreamState() *streamState {
	return &streamState{
		functionArgBufs: make(map[string]*strings.Builder),
		callIDByItemID:  make(map[string]string),
		nameByItemID:    make(map[string]string),
	}
}

// consume returns:
// - msg: message chunk (delta)
// - done: if this chunk ends the response
// - deltaOnly: whether it's a pure delta message (so no finalization)
func (s *streamState) consume(ev responses.ResponseStreamEventUnion) (msg *schema.Message, done bool, deltaOnly bool, err error) {
	switch v := ev.AsAny().(type) {
	case responses.ResponseErrorEvent:
		return nil, false, false, fmt.Errorf("openai stream error: %s (%s)", v.Message, v.Code)
	case responses.ResponseCreatedEvent:
		s.modelName = string(v.Response.Model)
		return nil, false, true, nil
	case responses.ResponseInProgressEvent:
		// ignore; model name can be here too
		s.modelName = string(v.Response.Model)
		return nil, false, true, nil
	case responses.ResponseTextDeltaEvent:
		if v.Delta == "" {
			return nil, false, true, nil
		}
		return &schema.Message{Role: schema.Assistant, Content: v.Delta}, false, true, nil
	case responses.ResponseReasoningTextDeltaEvent:
		if v.Delta == "" {
			return nil, false, true, nil
		}
		m := &schema.Message{Role: schema.Assistant, ReasoningContent: v.Delta}
		return m, false, true, nil
	case responses.ResponseOutputItemAddedEvent:
		// function call item appears here with call_id and name
		item := v.Item
		if item.Type == "function_call" {
			call := item.AsFunctionCall()
			s.callIDByItemID[item.ID] = call.CallID
			s.nameByItemID[item.ID] = call.Name
		}
		return nil, false, true, nil
	case responses.ResponseFunctionCallArgumentsDeltaEvent:
		if v.Delta == "" {
			return nil, false, true, nil
		}
		b := s.functionArgBufs[v.ItemID]
		if b == nil {
			b = &strings.Builder{}
			s.functionArgBufs[v.ItemID] = b
		}
		b.WriteString(v.Delta)
		return nil, false, true, nil
	case responses.ResponseFunctionCallArgumentsDoneEvent:
		// Finalize args: only emit ToolCalls when arguments are complete.
		callID := s.callIDByItemID[v.ItemID]
		name := s.nameByItemID[v.ItemID]
		if callID == "" {
			callID = v.ItemID
		}

		args := v.Arguments
		if args == "" {
			if b := s.functionArgBufs[v.ItemID]; b != nil {
				args = b.String()
			}
		}
		return &schema.Message{Role: schema.Assistant, ToolCalls: []schema.ToolCall{{
			ID:   callID,
			Type: "function",
			Function: schema.FunctionCall{
				Name:      name,
				Arguments: args,
			},
		}}}, false, true, nil
	case responses.ResponseCompletedEvent:
		// IMPORTANT: do not emit the full final assistant message content here.
		// The Responses streaming API already sends the assistant text as deltas
		// (ResponseTextDeltaEvent / ResponseReasoningTextDeltaEvent). Emitting the
		// final full message (resp.OutputText()) would cause downstream consumers
		// that concatenate chunks to duplicate output.
		return &schema.Message{
			Role: schema.Assistant,
			ResponseMeta: &schema.ResponseMeta{
				FinishReason: string(v.Response.Status),
				Usage:        toEinoTokenUsage(v.Response.Usage),
			},
		}, true, false, nil
	case responses.ResponseFailedEvent:
		return &schema.Message{Role: schema.Assistant, ResponseMeta: &schema.ResponseMeta{FinishReason: string(v.Response.Status)}}, true, false, nil
	case responses.ResponseIncompleteEvent:
		return &schema.Message{Role: schema.Assistant, ResponseMeta: &schema.ResponseMeta{FinishReason: string(v.Response.Status), Usage: toEinoTokenUsage(v.Response.Usage)}}, true, false, nil
	default:
		return nil, false, true, nil
	}
}
