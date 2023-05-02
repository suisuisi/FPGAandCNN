
# host
file(GLOB HOST_SRCS *.cc)

find_library(XRT_LIBRARIES NAMES xrt_core PATHS ${XILINX_XRT}/lib)

set(VITIS_INCLUDE_DIRS ${XILINX_XRT}/include ${XILINX_VIVADO}/include)
set(VITIS_LIBRARIES OpenCL pthread ${XRT_LIBRARIES})

add_executable(run_inference ${HOST_SRCS})
target_include_directories(run_inference PRIVATE ${DNNK_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${VITIS_INCLUDE_DIRS})
target_link_libraries(run_inference PRIVATE ${TORCH_LIBRARIES} ${VITIS_LIBRARIES})
target_compile_options(run_inference PRIVATE "-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"")
target_compile_features(run_inference PRIVATE cxx_std_14)

# xo
function (add_xo name top target sources platform)
  set(include_dirs ${DNNK_INCLUDE_DIRS})
  prepend_option("${include_dirs}" "-I" include_options)

  add_custom_target(
    ${name}_xo
    COMMAND v++ -g --compile --target ${target} --kernel ${top} --platform ${platform} --profile_kernel data:all:all:all --profile_kernel stall:all:all:all --temp_dir build_${name} --save-temps ${include_options} ${sources} -o ${name}.xo
    WORKING_DIRECTORY ${WORK_DIR}
    )
endfunction()

# xclbin
function (add_xclbin name top target ini_file platform)

  abs_path(${ini_file} abs_ini_file)

  set(top_func ${top})
  set(target_ini_file ${name}_${TARGET_BOARD}.ini)

  configure_file(${abs_ini_file} ${target_ini_file})
  
  add_custom_target(
    ${name}
    COMMAND v++ -g --link --target ${target} --platform ${platform} --config ${target_ini_file} --temp_dir build_${name} --save-temps ${name}.xo -o ${name}.xclbin
    WORKING_DIRECTORY ${WORK_DIR}
    )
endfunction()

function (add_xo_and_xclbin name top sources ini_file platform)

  add_xo(${name}_hw ${top} hw ${sources} ${platform})
  add_xo(${name}_hw_emu ${top} hw_emu ${sources} ${platform})
  add_xo(${name}_sw_emu ${top} sw_emu ${sources} ${platform})
  add_xclbin(${name}_hw ${top} hw ${ini_file} ${platform})
  add_xclbin(${name}_hw_emu ${top} hw_emu ${ini_file} ${platform})
  add_xclbin(${name}_sw_emu ${top} sw_emu ${ini_file} ${platform})
endfunction()
  
get_filename_component(inference_src ../tests/hls/inference/inference_hls.cc ABSOLUTE)

add_xo_and_xclbin(inference_top inference_top ${inference_src} link_${TARGET_BOARD}_template.ini ${VITIS_PLATFORM})
add_xo_and_xclbin(inference_dataflow inference_dataflow ${inference_src} link_${TARGET_BOARD}_template.ini ${VITIS_PLATFORM})
add_xo_and_xclbin(inference_with_local_buffer inference_with_local_buffer ${inference_src} link_${TARGET_BOARD}_template.ini ${VITIS_PLATFORM})
add_xo_and_xclbin(inference_pipelined_conv_v1 inference_pipelined_conv_v1 ${inference_src} link_${TARGET_BOARD}_template.ini ${VITIS_PLATFORM})
add_xo_and_xclbin(inference_pipelined_conv_v2 inference_pipelined_conv_v2 ${inference_src} link_${TARGET_BOARD}_template.ini ${VITIS_PLATFORM})
add_xo_and_xclbin(inference_unrolledx4_conv_v1 inference_unrolledx4_conv_v1 ${inference_src} link_${TARGET_BOARD}_template.ini ${VITIS_PLATFORM})
add_xo_and_xclbin(inference_unrolledx4_conv_v2 inference_unrolledx4_conv_v2 ${inference_src} link_${TARGET_BOARD}_template.ini ${VITIS_PLATFORM})
add_xo_and_xclbin(inference_final inference_final ${inference_src} link_${TARGET_BOARD}_template.ini ${VITIS_PLATFORM})
