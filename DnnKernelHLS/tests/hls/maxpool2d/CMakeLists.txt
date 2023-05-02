set(hls_src maxpool2d_hls.cc)
set(test_src maxpool2d_test.cc)

add_test_and_impl(maxpool2d_hls maxpool2d_hls HLS_SRC ${hls_src} TB_SRC ${test_src})
