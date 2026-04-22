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
	"testing"

	"github.com/cloudwego/eino/schema"
)

func TestNewChatModel_NilConfig(t *testing.T) {
	cm, err := NewChatModel(context.Background(), nil)
	if err == nil {
		t.Fatalf("expected error")
	}
	if cm != nil {
		t.Fatalf("expected nil model")
	}
}

func TestNewChatModel_Basic(t *testing.T) {
	cm, err := NewChatModel(context.Background(), &Config{APIKey: "test", Model: "gpt-4o-mini"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cm == nil {
		t.Fatalf("expected non-nil model")
	}
	if cm.GetType() != typ {
		t.Fatalf("expected type %q, got %q", typ, cm.GetType())
	}
	if !cm.IsCallbacksEnabled() {
		t.Fatalf("expected callbacks enabled")
	}
}

func TestToInputItems_ToolOutputString(t *testing.T) {
	items, err := toInputItems([]*schema.Message{{
		Role:       schema.Tool,
		ToolCallID: "call_1",
		Content:    "ok",
	}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
}
