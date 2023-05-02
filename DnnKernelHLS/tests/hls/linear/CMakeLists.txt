set(hls_src linear_hls.cc)
set(test_src linear_test.cc)

add_test_and_impl(linear_hls linear_hls HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DTOP_FUNC=linear_hls")
add_test_and_impl(linear_opt_2_hls linear_opt_2_hls HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DTOP_FUNC=linear_opt_2_hls")
add_test_and_impl(linear_opt_3_hls linear_opt_3_hls HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DTOP_FUNC=linear_opt_3_hls")
