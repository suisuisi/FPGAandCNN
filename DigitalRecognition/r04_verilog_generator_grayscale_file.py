# coding: utf-8
__author__ = 'Alex Kustov, IPPM RAS'


import os

gpu_use = 0
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from r03_find_optimal_bit_for_weights import get_optimal_bit_for_weights


def grayscale(directory, bit_size):
    file = open(directory + "grayscale.v", 'w')

    bit_sr_data = len(bin(((2**bit_size)-1)*64)) - 2
    square_size = 28

    file.write("module pre_v2(clk, rst_n, start, data, end_pre, output_data, x, y, i, j ,data_req);\n")
    file.write("  input clk;\n")
    file.write("  input rst_n;\n")
    file.write("  input [15:0] data;\n")
    file.write("  input start;\n")
    file.write("  output reg end_pre;\n")
    file.write("  output reg ["+str(bit_size-1)+":0] output_data;\n")
    file.write("  input [9:0] x, y;\n")
    file.write("  output reg [4:0] i, j;\n")
    file.write("  output data_req;\n")
    file.write("  \n")
    file.write("\n")
    for i in range(square_size):
        file.write("reg [{}:0] sr_data_{}_0;\n".format(bit_sr_data - 1, i))
        file.write("reg wr_sr_data_{};\n".format(i))
        file.write("wire [{}:0] output_data_{}_0;\n".format(bit_size - 1, i))
    file.write("\n")
    file.write("reg [{}:0] R;\n".format(bit_size-1))
    file.write("reg [{}:0] G;\n".format(bit_size-1))
    file.write("reg [{}:0] B;\n".format(bit_size-1))
    file.write("reg [17:0] gray;\n")
    file.write("  \n")
    file.write("always @(posedge clk or negedge rst_n)	\n")
    file.write("	begin\n")
    file.write("	if ( !rst_n )\n")
    file.write("		begin\n")
    file.write("			i = 5'b0;\n")
    file.write("			j = 5'b0;\n")
    for i in range(square_size):
        file.write("			sr_data_{}_0 = 30'd0;\n".format(i))
    file.write("			end_pre = 1'b0;\n")
    file.write("			gray = 18'd0;\n")
    file.write("		end\n")
    file.write("	else\n")
    file.write("    begin\n")
    file.write("      if (start)\n")
    file.write("        begin\n")
    file.write("			R = {}'d0;\n".format(bit_size))
    file.write("			R[{}] = data[15];\n".format(bit_size-2))
    file.write("			R[{}] = data[14];\n".format(bit_size-3))
    file.write("			R[{}] = data[13];\n".format(bit_size-4))
    file.write("			R[{}] = data[12];\n".format(bit_size-5))
    file.write("			R[{}] = data[11];\n".format(bit_size-6))
    file.write("\n")
    file.write("			G = {}'d0;\n".format(bit_size))
    file.write("			G[{}] = data[10];\n".format(bit_size-2))
    file.write("			G[{}] = data[9];\n".format(bit_size-3))
    file.write("			G[{}] = data[8];\n".format(bit_size-4))
    file.write("			G[{}] = data[7];\n".format(bit_size-5))
    file.write("			G[{}] = data[6];\n".format(bit_size-6))
    file.write("			G[{}] = data[5];\n".format(bit_size-7))
    file.write("\n")
    file.write("			B = {}'d0;\n".format(bit_size))
    file.write("			B[{}] = data[4];\n".format(bit_size-2))
    file.write("			B[{}] = data[3];\n".format(bit_size-3))
    file.write("			B[{}] = data[2];\n".format(bit_size-4))
    file.write("			B[{}] = data[1];\n".format(bit_size-5))
    file.write("			B[{}] = data[0];\n".format(bit_size-6))
    file.write("				  \n")
    file.write("\n")
    file.write("         gray = 3*B + 8*G + 5*R;\n")
    file.write("         gray = gray >> 4;\n")
    file.write("		 \n")
    for i in range(square_size):
        file.write("			if ((x>(10'd47+({}*8)))&&(x<(10'd56+({}*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) \n".format(i, i))
        file.write("				begin		\n")
        file.write("				  sr_data_{}_0 = sr_data_{}_0 + gray;\n".format(i, i))
        file.write("				end\n")
    for i in range(square_size):
        file.write("			if ((x==(10'd57+({}*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_{}_0; i={}; end\n".format(i, i, i))
        file.write("			if ((x>(10'd59+({}*8)))&&(x<(10'd65+({}*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_{}=1'b1; \n".format(i, i, i))
        file.write("			else wr_sr_data_{} = 1'b0;\n".format(i))
    file.write("			\n")
    file.write("			if ((x==10'd319) && (y==(10'd15+(j*8))))\n")
    file.write("				begin\n")
    file.write("					if (j >= 5'd27) \n")
    file.write("					begin	\n")
    file.write("					  //j = 5'd0;\n")
    file.write("					  end_pre = 1'b1;\n")
    file.write("					end\n")
    file.write("					else j = j + 1'b1;\n")
    for i in range(square_size):
        file.write("					sr_data_{}_0 = 30'd0;\n".format(i))
    file.write("				end\n")
    file.write("        end\n")
    file.write("		else\n")
    file.write("			   begin\n")
    file.write("						i=5'b0;\n")
    file.write("						j=5'b0;\n")
    for i in range(square_size):
        file.write("						sr_data_{}_0 = 30'd0;\n".format(i))
    file.write("						end_pre = 1'b0;\n")
    file.write("						gray = 18'd0;\n")
    file.write("				end\n")
    file.write("			\n")
    file.write("    end\n")
    file.write("end\n")
    file.write("\n")
    file.write("        \n")
    file.write("assign data_req = wr_sr_data_0")
    for i in range(1, square_size):
        file.write(" | wr_sr_data_{}".format(i))
    file.write(";\n")
    for i in range(square_size):
        file.write("assign output_data_{}_0 = sr_data_{}_0 >> 6;\n".format(i, i))
    file.write("\n")
    file.write("endmodule\n")

    file.close()


if __name__ == '__main__':

    # Where to store verilog, which converts RGB image from camera to 28x28 grayscale image
    output_directory = "./verilog/code/gray_28x28/"
    # Bit size of weights (including sign)
    bit_size = get_optimal_bit_for_weights() + 1

    print('Create verilog in directory: {}'.format(output_directory))
    print('Bit size: {}'.format(bit_size))

    print("Make grayscale file")
    grayscale(output_directory,bit_size)
