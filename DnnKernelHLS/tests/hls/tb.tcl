set mode [lindex $argv 2]
set name [lindex $argv 3]
set hls_srcs [lindex $argv 4]
set top [lindex $argv 5]
set chip_part [lindex $argv 6]
set cxxflags [lindex $argv 7]
set ldflags [lindex $argv 8]
set test_srcs [lindex $argv 9]
set test_args [lindex $argv 10]


open_project -reset ${name}

regsub "cxxflags=" $cxxflags {} cxxflags
regsub "ldflags=" $ldflags {} ldflags
set test_cxxflags "${cxxflags} -std=c++14 -fopenmp"

set_top ${top}
add_files ${hls_srcs} -cflags "${cxxflags}"

open_solution "solution1"
set_part ${chip_part}
create_clock -period 3.33 -name default

csynth_design

if {${mode} == "cosim"} {
    add_files -tb ${test_srcs} -cflags "${test_cxxflags}"
    cosim_design -trace_level port -ldflags "${ldflags}" -argv "${test_args}"
}

if {${mode} == "impl"} {
    export_design -flow impl -rtl verilog -format ip_catalog
}

if {${mode} == "xo"} {
    config_rtl -kernel_profile
    config_sdx -target xocc -profile true
    export_design -flow impl -rtl verilog -format ip_catalog -xo ${name}.xo
}

exit
