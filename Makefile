PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
CONFIG ?= config/config.yaml
VOCAB ?= config/vocabulary.yaml
OUTPUT_DIR ?= out
FILE ?=
DIR ?= samples
BATCH_SIZE ?= 64
LIMIT ?=

.PHONY: help cache run batch

.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "Audio Analyzer"
	@echo ""
	@echo "Targets:"
	@echo "  make cache                       Build label embedding cache"
	@echo "  make run FILE=<path>             Analyze one audio file"
	@echo "  make batch [DIR=<path>]          Analyze a directory of audio files"
	@echo ""
	@echo "Config:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  VOCAB=$(VOCAB)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo ""
	@echo "Examples:"
	@echo "  make cache"
	@echo "  make cache BATCH_SIZE=32 FORCE=1"
	@echo "  make run FILE=samples/example.wav"
	@echo "  make batch DIR=samples LIMIT=10"
	@echo ""

cache:
	$(PYTHON) scripts/build_label_cache.py \
		--config "$(CONFIG)" \
		--vocab "$(VOCAB)" \
		--batch-size "$(BATCH_SIZE)" \
		$(if $(FORCE),--force,)

run:
ifndef FILE
	$(error FILE is not set. Usage: make run FILE=samples/example.wav)
endif
	$(PYTHON) analyze.py "$(FILE)" \
		--config "$(CONFIG)" \
 		--vocab "$(VOCAB)" \
		--output-dir "$(OUTPUT_DIR)" \
		$(RUN_ARGS)

batch:
	$(PYTHON) batch_process.py "$(DIR)" \
		--config "$(CONFIG)" \
		--vocab "$(VOCAB)" \
		--output-dir "$(OUTPUT_DIR)" \
		$(if $(LIMIT),--limit "$(LIMIT)",) \
		$(BATCH_ARGS)
