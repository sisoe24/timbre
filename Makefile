POETRY ?= poetry
USE_POETRY ?= 0
PYTHON ?= $(if $(filter 1,$(USE_POETRY)),$(POETRY) run python,$(if $(wildcard .venv/bin/python),.venv/bin/python,python3))
TIMBRE ?= $(if $(filter 1,$(USE_POETRY)),$(POETRY) run timbre,$(PYTHON) timbre.py)
DOCKER ?= docker
DOCKER_IMAGE ?= timbre
DOCKER_IMAGE_ARM64 ?= $(DOCKER_IMAGE):arm64
DOCKER_IMAGE_AMD64 ?= $(DOCKER_IMAGE):amd64
DOCKER_WORKDIR ?= /workspace
DOCKER_HF_CACHE ?= $(CURDIR)/.hf-cache
DIST_DIR ?= dist
DOCKER_TAR_ARM64 ?= $(DIST_DIR)/timbre-arm64.tar
DOCKER_TAR_AMD64 ?= $(DIST_DIR)/timbre-amd64.tar
CONFIG ?= config/config.yaml
VOCAB ?= config/vocabulary.yaml
OUTPUT_DIR ?= out
FILE ?=
DIR ?= samples
BATCH_SIZE ?= 64
LIMIT ?=

.PHONY: help install lock cache run batch docker-build docker-run docker-export-arm64 docker-export-amd64 docker-export-all

container_path = $(if $(filter /%,$(1)),$(1),$(DOCKER_WORKDIR)/$(1))

.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "Audio Analyzer"
	@echo ""
	@echo "Targets:"
	@echo "  make install                     Install dependencies with Poetry"
	@echo "  make lock                        Refresh poetry.lock"
	@echo "  make cache                       Build or refresh the active vocabulary cache"
	@echo "  make run FILE=<path>             Analyze one audio file"
	@echo "  make batch [DIR=<path>]          Analyze a directory of audio files"
	@echo "  make docker-build                Build the Docker image"
	@echo "  make docker-run FILE=<path>      Analyze one audio file via Docker"
	@echo "  make docker-export-arm64         Build and export an Apple Silicon image tar"
	@echo "  make docker-export-amd64         Build and export an Intel/AMD image tar"
	@echo "  make docker-export-all           Export both Docker image tar files"
	@echo ""
	@echo "Config:"
	@echo "  USE_POETRY=$(USE_POETRY)"
	@echo "  POETRY=$(POETRY)"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  TIMBRE=$(TIMBRE)"
	@echo "  DOCKER=$(DOCKER)"
	@echo "  DOCKER_IMAGE=$(DOCKER_IMAGE)"
	@echo "  DOCKER_IMAGE_ARM64=$(DOCKER_IMAGE_ARM64)"
	@echo "  DOCKER_IMAGE_AMD64=$(DOCKER_IMAGE_AMD64)"
	@echo "  DOCKER_HF_CACHE=$(DOCKER_HF_CACHE)"
	@echo "  DIST_DIR=$(DIST_DIR)"
	@echo "  DOCKER_TAR_ARM64=$(DOCKER_TAR_ARM64)"
	@echo "  DOCKER_TAR_AMD64=$(DOCKER_TAR_AMD64)"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  VOCAB=$(VOCAB)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo ""
	@echo "Examples:"
	@echo "  make install"
	@echo "  make cache"
	@echo "  make cache BATCH_SIZE=32 FORCE=1"
	@echo "  make run FILE=samples/example.wav"
	@echo "  make run USE_POETRY=1 FILE=samples/example.wav"
	@echo "  make batch DIR=samples LIMIT=10"
	@echo "  make docker-build"
	@echo "  make docker-run FILE=samples/example.wav"
	@echo "  make docker-export-arm64"
	@echo "  make docker-export-amd64"
	@echo "  make docker-export-all"
	@echo "  poetry run timbre analyze samples/example.wav"
	@echo "  poetry run timbre batch samples"
	@echo "  poetry run timbre vocab cache --force"
	@echo ""

install:
	$(POETRY) install

lock:
	$(POETRY) lock

cache:
	$(TIMBRE) vocab cache \
		--config "$(CONFIG)" \
		--vocab "$(VOCAB)" \
		--batch-size "$(BATCH_SIZE)" \
		$(if $(FORCE),--force,)

run:
ifndef FILE
	$(error FILE is not set. Usage: make run FILE=samples/example.wav)
endif
	$(TIMBRE) analyze "$(FILE)" \
		--config "$(CONFIG)" \
		--vocab "$(VOCAB)" \
		--output-dir "$(OUTPUT_DIR)" \
		$(RUN_ARGS)

batch:
	$(TIMBRE) batch "$(DIR)" \
		--config "$(CONFIG)" \
		--vocab "$(VOCAB)" \
		--output-dir "$(OUTPUT_DIR)" \
		$(if $(LIMIT),--limit "$(LIMIT)",) \
		$(BATCH_ARGS)

docker-build:
	$(DOCKER) build -t "$(DOCKER_IMAGE)" .

docker-run:
ifndef FILE
	$(error FILE is not set. Usage: make docker-run FILE=samples/example.wav)
endif
	mkdir -p "$(OUTPUT_DIR)" "$(DOCKER_HF_CACHE)"
	$(DOCKER) run --rm \
		-v "$(CURDIR):$(DOCKER_WORKDIR)" \
		-v "$(DOCKER_HF_CACHE):/root/.cache/huggingface" \
		"$(DOCKER_IMAGE)" analyze "$(call container_path,$(FILE))" \
		--config "$(call container_path,$(CONFIG))" \
		--vocab "$(call container_path,$(VOCAB))" \
		--output-dir "$(call container_path,$(OUTPUT_DIR))" \
		$(RUN_ARGS)

docker-export-arm64:
	mkdir -p "$(DIST_DIR)"
	$(DOCKER) build -t "$(DOCKER_IMAGE_ARM64)" .
	$(DOCKER) save -o "$(DOCKER_TAR_ARM64)" "$(DOCKER_IMAGE_ARM64)"

docker-export-amd64:
	mkdir -p "$(DIST_DIR)"
	$(DOCKER) buildx build --platform linux/amd64 --load -t "$(DOCKER_IMAGE_AMD64)" .
	$(DOCKER) save -o "$(DOCKER_TAR_AMD64)" "$(DOCKER_IMAGE_AMD64)"

docker-export-all: docker-export-arm64 docker-export-amd64
