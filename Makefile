POETRY ?= poetry
USE_POETRY ?= 0
PYTHON ?= $(if $(filter 1,$(USE_POETRY)),$(POETRY) run python,$(if $(wildcard .venv/bin/python),.venv/bin/python,python3))
CONFIG ?= config/config.yaml
VOCAB ?= config/vocabulary.yaml
OUTPUT_DIR ?= out
FILE ?=
DIR ?= samples
BATCH_SIZE ?= 64
LIMIT ?=

.PHONY: help install lock cache run batch

.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "Audio Analyzer"
	@echo ""
	@echo "Targets:"
	@echo "  make install                     Install dependencies with Poetry"
	@echo "  make lock                        Refresh poetry.lock"
	@echo "  make cache                       Build label embedding cache"
	@echo "  make run FILE=<path>             Analyze one audio file"
	@echo "  make batch [DIR=<path>]          Analyze a directory of audio files"
	@echo ""
	@echo "Config:"
	@echo "  USE_POETRY=$(USE_POETRY)"
	@echo "  POETRY=$(POETRY)"
	@echo "  PYTHON=$(PYTHON)"
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
	@echo "  poetry run timbre analyze samples/example.wav"
	@echo "  poetry run timbre batch samples"
	@echo "  poetry run timbre cache --force"
	@echo ""

install:
	$(POETRY) install

lock:
	$(POETRY) lock

cache:
	$(PYTHON) timbre.py cache \
		--config "$(CONFIG)" \
		--vocab "$(VOCAB)" \
		--batch-size "$(BATCH_SIZE)" \
		$(if $(FORCE),--force,)

run:
ifndef FILE
	$(error FILE is not set. Usage: make run FILE=samples/example.wav)
endif
	$(PYTHON) timbre.py analyze "$(FILE)" \
		--config "$(CONFIG)" \
		--vocab "$(VOCAB)" \
		--output-dir "$(OUTPUT_DIR)" \
		$(RUN_ARGS)

batch:
	$(PYTHON) timbre.py batch "$(DIR)" \
		--config "$(CONFIG)" \
		--vocab "$(VOCAB)" \
		--output-dir "$(OUTPUT_DIR)" \
		$(if $(LIMIT),--limit "$(LIMIT)",) \
		$(BATCH_ARGS)
