set(hls_src conv2d_hls.cc)
set(test_src conv2d_test.cc)

add_test_and_impl(conv2d_hls conv2d_hls HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DTOP_FUNC=conv2d_hls")
add_test_and_impl(conv2d_pipelined_v1_hls conv2d_pipelined_v1_hls HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DTOP_FUNC=conv2d_pipelined_v1_hls")
add_test_and_impl(conv2d_pipelined_v2_hls conv2d_pipelined_v2_hls HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DTOP_FUNC=conv2d_pipelined_v2_hls")
add_test_and_impl(conv2d_unrolled_v1_2_hls conv2d_unrolled_v1_2_hls HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DTOP_FUNC=conv2d_unrolled_v1_2_hls")
add_test_and_impl(conv2d_unrolled_v1_3_hls conv2d_unrolled_v1_3_hls HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DTOP_FUNC=conv2d_unrolled_v1_3_hls")
add_test_and_impl(conv2d_unrolled_v2_2_2_hls conv2d_unrolled_v2_2_2_hls HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DTOP_FUNC=conv2d_unrolled_v2_2_2_hls")
add_test_and_impl(conv2d_unrolled_v2_2_3_hls conv2d_unrolled_v2_2_3_hls HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DTOP_FUNC=conv2d_unrolled_v2_2_3_hls")
add_test_and_impl(conv2d_unrolled_v2_3_2_hls conv2d_unrolled_v2_3_2_hls HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DTOP_FUNC=conv2d_unrolled_v2_3_2_hls")
add_test_and_impl(conv2d_unrolled_v2_3_3_hls conv2d_unrolled_v2_3_3_hls HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DTOP_FUNC=conv2d_unrolled_v2_3_3_hls")

