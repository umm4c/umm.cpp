find_program(UMM_GIT git)

function(umm_revision directory prefix)
    set(commit "unknown")
    set(dirty "true")
    if(UMM_GIT)
        execute_process(COMMAND "${UMM_GIT}" -c safe.directory=${directory}
                        -C "${directory}" rev-parse HEAD
                        OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET RESULT_VARIABLE status)
        if(status EQUAL 0)
            set(commit "${revision}")
            execute_process(COMMAND "${UMM_GIT}" -c safe.directory=${directory}
                            -C "${directory}" status --porcelain --untracked-files=normal
                            OUTPUT_VARIABLE changes ERROR_QUIET RESULT_VARIABLE status)
            if(status EQUAL 0 AND changes STREQUAL "")
                set(dirty "false")
            endif()
        endif()
    endif()
    set(${prefix}_COMMIT "${commit}" PARENT_SCOPE)
    set(${prefix}_DIRTY "${dirty}" PARENT_SCOPE)
endfunction()

umm_revision("${UMM_SOURCE_DIR}" UMM)
umm_revision("${UMM_SOURCE_DIR}/third_party/llama.cpp" UMM_LLAMA)
umm_revision("${UMM_SOURCE_DIR}/third_party/stable-diffusion.cpp" UMM_SD)
set(content "#pragma once\n#define UMM_BUILD_COMMIT \"${UMM_COMMIT}\"\n#define UMM_BUILD_DIRTY ${UMM_DIRTY}\n#define UMM_LLAMA_COMMIT \"${UMM_LLAMA_COMMIT}\"\n#define UMM_LLAMA_DIRTY ${UMM_LLAMA_DIRTY}\n#define UMM_SD_COMMIT \"${UMM_SD_COMMIT}\"\n#define UMM_SD_DIRTY ${UMM_SD_DIRTY}\n")
if(EXISTS "${UMM_IDENTITY_HEADER}")
    file(READ "${UMM_IDENTITY_HEADER}" previous)
endif()
if(NOT previous STREQUAL content)
    get_filename_component(output_dir "${UMM_IDENTITY_HEADER}" DIRECTORY)
    file(MAKE_DIRECTORY "${output_dir}")
    file(WRITE "${UMM_IDENTITY_HEADER}" "${content}")
endif()
