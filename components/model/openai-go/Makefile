export GO111MODULE=on
export GOFLAGS=-mod=vendor

# Build/test tags. Defaults to enabling github.com/negrel/assert assertions.
# Override with e.g.:
#   make TAGS=
#   make TAGS="assert,other"
TAGS ?= assert
GO_TAG_FLAGS :=
ifneq ($(strip $(TAGS)),)
GO_TAG_FLAGS := -tags $(TAGS)
endif

.PHONY: vendor
vendor: vendor/modules.txt

vendor/modules.txt: go.mod go.sum
	# Ensure we can (re)generate vendor/ even when GOFLAGS forces -mod=vendor.
	GOFLAGS=-mod=mod go mod download
	GOFLAGS=-mod=mod go mod vendor

RACE ?= 1
TESTFLAGS ?=
ifeq ($(RACE),1)
TESTFLAGS += -race
endif

.PHONY: test
test: vendor
	go test $(GO_TAG_FLAGS) $(TESTFLAGS) ./...

unit-tests.xml: vendor FORCE
	gotestsum --junitfile unit-tests.xml -- $(GO_TAG_FLAGS) $(TESTFLAGS) ./...

.PHONY: lint
lint:
	golangci-lint run ./...

.PHONY: clean
clean:
	rm -f unit-tests.xml

.PHONY: deps
deps:
	rm -rf go.mod go.sum vendor
	go mod init github.com/cloudwego/eino-ext/components/model/openai-go
	go mod edit -go=1.22.3
	GOPROXY=direct go mod tidy
	go mod vendor

FORCE:
