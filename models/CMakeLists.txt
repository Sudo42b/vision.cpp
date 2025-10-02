# Download models used in tests (happens only if VISP_TESTS is enabled)

message(STATUS "Checking for models/MobileSAM-F16.gguf")
file(DOWNLOAD
  "https://huggingface.co/Acly/MobileSAM-GGUF/resolve/main/MobileSAM-F16.gguf"
  ${CMAKE_CURRENT_LIST_DIR}/MobileSAM-F16.gguf
  EXPECTED_HASH "SHA256=b546366475e3ad744bb2eaf7634df88e9aaf25f6622797d2de300f5a530831f7"
  SHOW_PROGRESS
)
message(STATUS "Checking for models/BiRefNet-lite-F16.gguf")
file(DOWNLOAD
  "https://huggingface.co/Acly/BiRefNet-GGUF/resolve/main/BiRefNet-lite-F16.gguf"
  ${CMAKE_CURRENT_LIST_DIR}/BiRefNet-lite-F16.gguf
  EXPECTED_HASH "SHA256=7b5397a2c98d66677f8f74317774bbeac49dbb321b8a3dc744af913db71d4fa5"
  SHOW_PROGRESS
)
message(STATUS "Checking for models/MIGAN-512-places2-F16.gguf")
file(DOWNLOAD
  "https://huggingface.co/Acly/MIGAN-GGUF/resolve/main/MIGAN-512-places2-F16.gguf"
  ${CMAKE_CURRENT_LIST_DIR}/MIGAN-512-places2-F16.gguf
  EXPECTED_HASH "SHA256=3e47592bf716d0dc306f8dc02d4476cfcdaf2c055fa3c3c8e0ced4db775eb64b"
  SHOW_PROGRESS
)
message(STATUS "Checking for models/RealESRGAN-x4plus_anime-6B-F16.gguf")
file(DOWNLOAD
  "https://huggingface.co/Acly/Real-ESRGAN-GGUF/resolve/main/RealESRGAN-x4plus_anime-6B-F16.gguf"
  ${CMAKE_CURRENT_LIST_DIR}/RealESRGAN-x4plus_anime-6B-F16.gguf
  EXPECTED_HASH "SHA256=730469c5a2269cdef96d0d58aacf87bcf25d7a0d92256685808e6cdce0675c09"
  SHOW_PROGRESS
)