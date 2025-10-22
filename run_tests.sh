#!/bin/bash

# 현재 프로젝트의 라이브러리 경로를 우선으로 설정
# Remove any external ggml paths that might shadow the local build libs
if [ -n "$LD_LIBRARY_PATH" ]; then
	# split LD_LIBRARY_PATH by ':' and remove any entries that match the external ggml path
	IFS=':' read -ra __parts <<< "$LD_LIBRARY_PATH"
	__new=""
	for __p in "${__parts[@]}"; do
		# skip entries that point to the ggml-python package lib dir
		case "${__p}" in
			*ggml-python/ggml/lib*) continue ;; # drop
			"") continue ;;
			*)
				if [ -z "${__new}" ]; then
					__new="${__p}"
				else
					__new="${__new}:${__p}"
				fi
				;;
		esac
	done
	LD_LIBRARY_PATH="${__new}"
fi

# Prepend the project's build/lib so it is searched first by the dynamic loader
export LD_LIBRARY_PATH="$(pwd)/build/lib${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
source ./.venv/bin/activate
# pytest 실행
uv run pytest "$@"