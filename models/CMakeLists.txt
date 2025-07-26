# Download models used in tests (happens only if VISP_TESTS is enabled)

message(STATUS "Checking for models/MobileSAM-F16.gguf")
file(DOWNLOAD
  "https://huggingface.co/Acly/MobileSAM-GGUF/resolve/main/MobileSAM-F16.gguf"
  ${CMAKE_CURRENT_LIST_DIR}/MobileSAM-F16.gguf
  EXPECTED_HASH "SHA256=1e392f58a0e518b7e1e9e5a43403ff0c6d001aeefa6f4e4d2bdf60f7bbe6e4f2"
  SHOW_PROGRESS
)
message(STATUS "Checking for models/BiRefNet-lite-F16.gguf")
file(DOWNLOAD
  "https://huggingface.co/Acly/BiRefNet-GGUF/resolve/main/BiRefNet-lite-F16.gguf"
  ${CMAKE_CURRENT_LIST_DIR}/BiRefNet-lite-F16.gguf
  EXPECTED_HASH "SHA256=f038843ea7c44a859491df96c7b36815143f7de77b13cbfc0dae5f6eae863fb5"
  SHOW_PROGRESS
)
message(STATUS "Checking for models/MIGAN-512-places2-F16.gguf")
file(DOWNLOAD
  "https://huggingface.co/Acly/MIGAN-GGUF/resolve/main/MIGAN-512-places2-F16.gguf"
  ${CMAKE_CURRENT_LIST_DIR}/MIGAN-512-places2-F16.gguf
  EXPECTED_HASH "SHA256=c9f241e96fb5a791f9494fc7d4c2dd793297ae95f05b8423f547d19bea465b81"
  SHOW_PROGRESS
)
message(STATUS "Checking for models/RealESRGAN-x4plus_anime-6B-F16.gguf")
file(DOWNLOAD
  "https://huggingface.co/Acly/Real-ESRGAN-GGUF/resolve/main/RealESRGAN-x4plus_anime-6B-F16.gguf"
  ${CMAKE_CURRENT_LIST_DIR}/RealESRGAN-x4plus_anime-6B-F16.gguf
  EXPECTED_HASH "SHA256=b741e68720d7ad6251dee2120bf7579ef816ea16da18299b39f6cbcb0e13ecf0"
  SHOW_PROGRESS
)