include(CMakeParseArguments)

set(TB_TCL ${CMAKE_CURRENT_SOURCE_DIR}/tb.tcl)

macro(abs_path files abs_paths)
  foreach(f ${files})
    list(APPEND ${abs_paths} ${CMAKE_CURRENT_SOURCE_DIR}/${f})
  endforeach()
endmacro()

macro(append_include_option src dst)
  foreach(p ${src})
    list(APPEND ${dst} "includepath=${p}")
  endforeach()
endmacro()

macro(list2str l str)
  string(REPLACE ";" " " ${str} "${l}")
endmacro()

macro(prepend_option srcs option dsts)
  foreach(src ${srcs})
    list(APPEND ${dsts} "${option}${src}")
  endforeach()
endmacro()


function(add_csim name)
  cmake_parse_arguments(ARG "" "" "HLS_SRC;TB_SRC;CXXFLAGS" ${ARGN})

  add_executable(${name} ${ARG_HLS_SRC} ${ARG_TB_SRC})

  # For checking c++ standards
  set_source_files_properties("${ARG_TB_SRC}" PROPERTIES COMPILE_FLAGS "-std=c++14")
  set_source_files_properties("${ARG_HLS_SRC}" PROPERTIES COMPILE_FLAGS "-std=c++98")

  target_include_directories(${name} PRIVATE ${DNNK_INCLUDE_DIRS} ${VHLS_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})
  target_compile_options(${name} PRIVATE ${DNNK_CXX_FLAGS} ${ARG_CXXFLAGS})
  target_link_libraries(${name} PRIVATE ${TORCH_LIBRARIES} ${GTEST_LIBRARIES})

  add_test(
    NAME ${name}
    COMMAND ${name} ${ARG_UNPARSED_ARGUMENTS}
    )

endfunction()

function(add_cosim name top)
  cmake_parse_arguments(ARG "" "" "HLS_SRC;TB_SRC;CXXFLAGS" ${ARGN})
 
  abs_path(${ARG_HLS_SRC} abs_hls_srcs)
  abs_path(${ARG_TB_SRC} abs_tb_srcs)
  
  set(include_dirs ${DNNK_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
  prepend_option("${include_dirs}" "-I" include_options)
  
  set(library_dirs ${TORCH_LIBRARY_DIRS})
  prepend_option("${library_dirs}" "-L" library_dir_options)
  prepend_option("${library_dirs}" "-Wl,-rpath," rpath_options)

  set(libraries "torch" "c10" "gtest" "pthread")
  prepend_option("${libraries}" "-l" librariy_options)

  set(cxxflags ${ARG_CXXFLAGS} ${include_options})
  list2str("${cxxflags}" cxxflags_str)

  set(ldflags ${library_dir_options} ${librariy_options} ${rpath_options})
  list2str("${ldflags}" ldflags_str)
  
  add_test(
    NAME ${name}
    COMMAND vivado_hls -f ${TB_TCL} "cosim" ${name} "${abs_hls_srcs}" ${top} ${CHIP_PART} "cxxflags=${cxxflags_str}" "ldflags=${ldflags_str}" "${abs_tb_srcs}" "${ARG_UNPARSED_ARGUMENTS}"
    WORKING_DIRECTORY ${WORK_DIR}
    )
endfunction()

function(add_impl name top)
  cmake_parse_arguments(ARG "" "" "HLS_SRC;CXXFLAGS" ${ARGN})

  abs_path(${ARG_HLS_SRC} abs_hls_srcs)

  set(include_dirs ${DNNK_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
  prepend_option("${include_dirs}" "-I" include_options)
  set(cxxflags ${ARG_CXXFLAGS} ${include_options})
  list2str("${cxxflags}" cxxflags_str)

  add_custom_target(
    ${name}
    COMMAND vivado_hls -f ${TB_TCL} "impl" ${name} "${abs_hls_srcs}" ${top} ${CHIP_PART} "cxxflags=${cxxflags_str}" ${ARG_UNPARSED_ARGUMENTS}
    WORKING_DIRECTORY ${WORK_DIR}
    )
endfunction()

function(add_test_and_impl name top)
  cmake_parse_arguments(ARG "" "" "HLS_SRC;TB_SRC;CXXFLAGS" ${ARGN})

  add_csim(${name}_csim CXXFLAGS ${ARG_CXXFLAGS} HLS_SRC ${ARG_HLS_SRC} TB_SRC ${ARG_TB_SRC})
  add_cosim(${name}_cosim ${top} CXXFLAGS ${ARG_CXXFLAGS} HLS_SRC ${ARG_HLS_SRC} TB_SRC ${ARG_TB_SRC})
  add_impl(${name}_impl ${top} CXXFLAGS ${ARG_CXXFLAGS} HLS_SRC ${ARG_HLS_SRC})
endfunction()

add_subdirectory(relu)
add_subdirectory(conv2d)
add_subdirectory(maxpool2d)
add_subdirectory(linear)
add_subdirectory(inference)
