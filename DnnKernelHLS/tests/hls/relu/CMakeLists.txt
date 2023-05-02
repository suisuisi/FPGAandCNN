set(hls_src relu_hls.cc)
set(test_src relu_test.cc)

add_test_and_impl(relu_hls relu_hls HLS_SRC ${hls_src} TB_SRC ${test_src})
