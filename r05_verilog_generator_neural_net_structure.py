# coding: utf-8
__author__ = 'Alex Kustov, IPPM RAS'


import os

gpu_use = 0
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

import numpy as np
from a01_model_low_weights_digit_detector import keras_model_low_weights_digit_detector
from r03_find_optimal_bit_for_weights import get_optimal_bit_for_weights


def border(directory, razmer):
    file = open(directory+"border.v",'w')

    bit_matrix = len(bin(razmer))-2
    bit_matrix_2=len(bin(razmer*razmer))-2

    file.write("module border(\n")
    file.write("    input clk, go,\n")
    file.write("    input ["+str(bit_matrix_2-1)+":0] i,\n")
    file.write("    input ["+str(bit_matrix-1)+":0] matrix,\n")
    file.write("    output reg [1:0] prov\n")
    file.write(");\n")
    file.write("reg ["+str(bit_matrix_2)+":0] j;\n")
    file.write("always @(posedge clk)\n")
    file.write("begin\n")
    file.write("    if (go == 1)\n")
    file.write("    begin\n")
    file.write("        prov = 0;\n")
    file.write("        for (j = 1'b1; j <= matrix; j = j + 1'b1)\n")
    file.write("        begin\n")
    file.write("            if ((i == j*matrix-1'b1) && (prov != 2'b10))\n")
    file.write("                prov = 2'b10;\n")
    file.write("            if (((i == 0) || (i == j*matrix)) && (prov != 2'b11))\n")
    file.write("                prov = 2'b11;\n")
    file.write("        end\n")
    file.write("    end\n")
    file.write("    else\n")
    file.write("        prov = 0;\n")
    file.write("end\n")
    file.write("endmodule")

    file.close()


def maxpooling(directory, razmer, num_conv):
    file = open(directory + "maxpooling.v", 'w')

    bit_matrix=len(bin(razmer))-2
    bit_matrix_2=len(bin(razmer*razmer))-2

    file.write("module maxp(clk,maxp_en,memstartp,memstartzap,read_addressp,write_addressp,re,we,qp,dp,STOP,matrix2,matrix);\n\n")
    file.write("parameter SIZE_1=0;\n")
    file.write("parameter SIZE_2=0;\n")
    file.write("parameter SIZE_3=0;\n")
    file.write("parameter SIZE_4=0;\n")
    file.write("parameter SIZE_address_pix=0;\n\n")
    file.write("input clk,maxp_en;\n")
    file.write("output reg STOP;\n")
    file.write("input [SIZE_address_pix-1:0] memstartp,memstartzap;\n")
    file.write("output reg [SIZE_address_pix-1:0] read_addressp,write_addressp;\n")
    file.write("output reg re,we;\n")
    file.write("input signed [SIZE_"+str(num_conv)+"-1:0] qp;\n")
    file.write("output reg signed [SIZE_"+str(num_conv)+"-1:0] dp;\n")
    file.write("input ["+str(bit_matrix-1)+":0] matrix;\n")
    file.write("input ["+str(bit_matrix_2-1)+":0] matrix2;\n\n")
    file.write("reg ["+str(bit_matrix_2-1)+":0] i;\n")
    file.write("reg [" + str(bit_matrix_2 - 1) + ":0] j;\n")
    file.write("reg [SIZE_"+str(num_conv)+"-1:0] buff;\n")
    file.write("reg [2:0] marker;\n")         #4 состояния автомата/выборка из 4 чисел
    file.write("wire ["+str(bit_matrix_2 - 1)+":0] i_wr,i_read;\n")
    file.write("initial i=0;\n")
    file.write("initial j=0;\n")
    file.write("initial marker=0;\n")
    file.write("always @(posedge clk)\n")
    file.write("begin\n")
    file.write("if (maxp_en==1)\n")
    file.write("    begin\n")
    file.write("    case (marker)\n")
    file.write("        0: begin read_addressp=memstartp+i_read; re=1;\n")
    file.write("                 if ((i!=0)||(j!=0))\n")
    file.write("                 begin\n")

    for i in range(num_conv):
        one="SIZE_"+str(i+1)
        if (i==0): two="0"
        else: two="SIZE_"+str(i)
        file.write("                 if (qp["+one+"-1:"+two+"]>buff["+one+"-1:"+two+"]) buff["+one+"-1:"+two+"]=qp["+one+"-1:"+two+"];\n")

    file.write("                 end\n")
    file.write("           end\n")
    file.write("        1: begin read_addressp=memstartp+i_read+1;\n")
    file.write("                 if ((i!=0)||(j!=0))\n")
    file.write("                 begin\n")

    for i in range(num_conv):
        one="SIZE_"+str(i+1)
        if (i==0): two="0"
        else: two="SIZE_"+str(i)
        file.write("                 if (qp["+one+"-1:"+two+"]>buff["+one+"-1:"+two+"]) dp["+one+"-1:"+two+"]=qp["+one+"-1:"+two+"]; else dp["+one+"-1:"+two+"]=buff["+one+"-1:"+two+"];\n")

    file.write("                 write_addressp=memstartzap+i_wr-1;\n")
    file.write("                 we=1;\n")
    file.write("                 end\n")
    file.write("           end\n")
    file.write("        2: begin read_addressp=memstartp+i_read+matrix; buff=qp; we=0; end\n")
    file.write("        3: begin read_addressp=memstartp+i_read+matrix+1;\n")

    for i in range(num_conv):
        one="SIZE_"+str(i+1)
        if (i==0): two="0"
        else: two="SIZE_"+str(i)
        file.write("                 if (qp["+one+"-1:"+two+"]>buff["+one+"-1:"+two+"]) buff["+one+"-1:"+two+"]=qp["+one+"-1:"+two+"];\n")

    file.write("                 if (i!=matrix-2) begin i=i+2; if (i_read==matrix2) STOP=1; end else begin i=0; j=j+1;  end\n")
    file.write("           end\n")
    file.write("        default: $display(\"Check case MaxPooling\");\n")
    file.write("        endcase\n")
    file.write("        if (marker!=3) marker=marker+1; else marker=0;")
    file.write("    end\n")
    file.write("else\n")
    file.write("    begin\n")
    file.write("        STOP=0;\n")
    file.write("        re=0;\n")
    file.write("        we=0;\n")
    file.write("        i=0;\n")
    file.write("        j=0;\n")
    file.write("        marker=0;\n")
    file.write("    end\n")
    file.write("end\n")
    file.write("assign i_wr=(i>>1)+j*(matrix>>1);\n")
    file.write("assign i_read=i+(matrix+matrix)*j;\n")
    file.write("endmodule\n")

    file.close()


def RAM(directory, max_weights_per_layer, num_conv):
    file = open(directory + "RAM.v", 'w')

    file.write("module RAM(qp,qtp,qw,dp,dtp,dw,write_addressp,read_addressp,write_addresstp,read_addresstp,write_addressw,read_addressw,we_p,we_tp,we_w,re_p,re_tp,re_w,clk);\n")
    file.write("parameter picture_size=0;\n")
    file.write("parameter SIZE_1=0;\n")
    file.write("parameter SIZE_2=0;\n")
    file.write("parameter SIZE_4=0;\n")
    file.write("parameter SIZE_9=0;\n")
    file.write("parameter SIZE_address_pix=0;\n")
    file.write("parameter SIZE_address_pix_t=0;\n")
    file.write("parameter SIZE_address_wei=0;\n\n")
    file.write("output reg signed [SIZE_"+str(num_conv)+"-1:0] qp;       //read data\n")
    file.write("output reg signed [(SIZE_2)*"+str(num_conv)+"-1:0] qtp;       //read data\n")
    file.write("output reg signed [SIZE_9-1:0] qw;      //read weight\n")
    file.write("input signed [SIZE_"+str(num_conv)+"-1:0] dp;   //write data\n")
    file.write("input signed [(SIZE_2)*"+str(num_conv)+"-1:0] dtp;   //write data\n")
    file.write("input signed [SIZE_9-1:0] dw;   //write weight\n")
    file.write("input [SIZE_address_pix-1:0] write_addressp, read_addressp;\n")
    file.write("input [SIZE_address_pix_t-1:0] write_addresstp, read_addresstp;\n")
    file.write("input [SIZE_address_wei-1:0] write_addressw, read_addressw;\n")
    file.write("input we_p,we_tp,we_w,re_p,re_tp,re_w,clk;\n\n")
    file.write("reg signed [SIZE_"+str(num_conv)+"-1:0] mem [0:picture_size*picture_size*"+str(int(8/num_conv))+"+picture_size*picture_size-1];\n")
    file.write("reg signed [(SIZE_2)*"+str(num_conv)+"-1:0] mem_t [0:picture_size*picture_size*4-1];\n")
    file.write("reg signed [SIZE_9-1:0] weight [0:"+str(max_weights_per_layer)+"];\n")
    file.write("always @ (posedge clk)\n")
    file.write("    begin\n")
    file.write("        if (we_p) mem[write_addressp] <= dp;\n")
    file.write("		if (we_tp) mem_t[write_addresstp] <= dtp;\n")
    file.write("		if (we_w) weight[write_addressw] <= dw;\n")
    file.write("    end\n")
    file.write("always @ (posedge clk)\n")
    file.write("    begin\n")
    file.write("        if (re_p) qp <= mem[read_addressp];\n")
    file.write("		if (re_tp) qtp <= mem_t[read_addresstp];\n")
    file.write("        if (re_w) qw <= weight[read_addressw];\n")
    file.write("    end\n")
    file.write("endmodule")

    file.close()


def dense(directory, in_dense_razmer, out_dense_razmer, num_conv):
    file = open(directory + "dense.v", 'w')

    bit_in = len(bin(in_dense_razmer)) - 2
    bit_out = len(bin(out_dense_razmer)) - 2
    bit_in_dense_razmer = len(bin(in_dense_razmer)) - 2
    Y=''
    w=''
    p=''
    Ypl=''
    Y_use=''
    Y_use_pl=''
    for i in range(num_conv):
        Y=Y+" Y"+str(i+1)+","
        Y_use=Y_use+" Y"+str(i+1)+"_use,"
        Y_use_pl = Y_use_pl + " Y" + str(i + 1) + "_use+"
        Ypl = Ypl + " (Y" + str(i + 1) + "_use?Y" + str(i + 1) + ":0)+"
        w=w+" w"+str(i+1)+"1, w"+str(i+1)+"2, w"+str(i+1)+"3, w"+str(i+1)+"4, w"+str(i+1)+"5, w"+str(i+1)+"6, w"+str(i+1)+"7, w"+str(i+1)+"8, w"+str(i+1)+"9,"
        p=p+" p"+str(i+1)+"1, p"+str(i+1)+"2, p"+str(i+1)+"3, p"+str(i+1)+"4, p"+str(i+1)+"5, p"+str(i+1)+"6, p"+str(i+1)+"7, p"+str(i+1)+"8, p"+str(i+1)+"9,"

    file.write("module dense(clk, dense_en, STOP, in, out, we, re_p, re_w, read_addressp, read_addressw, write_addressp, memstartp, memstartzap, qp, qw, res,"+Y+w+p+" go, nozero, in_dense);\n\n")
    file.write("parameter num_conv=0;\n")
    file.write("parameter SIZE_1=0;\n")
    file.write("parameter SIZE_2=0;\n")
    file.write("parameter SIZE_3=0;\n")
    file.write("parameter SIZE_4=0;\n")
    file.write("parameter SIZE_5=0;\n")
    file.write("parameter SIZE_6=0;\n")
    file.write("parameter SIZE_7=0;\n")
    file.write("parameter SIZE_8=0;\n")
    file.write("parameter SIZE_9=0;\n")
    file.write("parameter SIZE_address_pix=0;\n")
    file.write("parameter SIZE_address_wei=0;\n\n")
    file.write("input clk,dense_en;\n")
    file.write("output reg STOP;\n")
    file.write("input ["+str(bit_in-1)+":0] in;\n")
    file.write("input ["+str(bit_out-1)+":0] out;\n")
    file.write("output reg we,re_p,re_w;\n")
    file.write("output reg [SIZE_address_pix-1:0] read_addressp;\n")
    file.write("output reg [SIZE_address_wei-1:0] read_addressw;\n")
    file.write("output reg [SIZE_address_pix-1:0] write_addressp;\n")
    file.write("input [SIZE_address_pix-1:0] memstartp,memstartzap;\n")
    file.write("input signed [SIZE_"+str(num_conv)+"-1:0] qp;\n")
    file.write("input signed [SIZE_9-1:0] qw;\n")
    file.write("output reg signed [SIZE_"+str(num_conv)+"-1:0] res;\n")
    file.write("input signed [SIZE_1+SIZE_1-2:0]"+Y[:-1]+";\n")
    file.write("output reg signed [SIZE_1-1:0]"+w[:-1]+";\n")
    file.write("output reg signed [SIZE_1-1:0]"+p[:-1]+";\n")
    file.write("output reg go;\n")
    file.write("input ["+str(bit_in_dense_razmer-1)+":0] in_dense;\n")
    file.write("input nozero;\n\n")
    file.write("reg [3:0] marker;\n")
    file.write("reg [6:0] lvl;\n")
    file.write("reg [8:0] i;\n")
    file.write("reg [8:0] j;\n")

    bit_sh=len(bin(num_conv)) - 3
    if (bit_sh>0): sh="["+str(bit_sh)+":0]"
    else: sh=''

    file.write("wire "+Y_use[:-1]+";\n")
    file.write("reg "+sh+" sh;\n")
    file.write("reg signed [(SIZE_2)*"+str(num_conv)+"-1:0] dp;\n")
    file.write("reg signed [SIZE_1-1:0] dp_shift;\n")
    file.write("reg signed [SIZE_1-1+1:0]dp_check;\n\n")
    file.write("always @(posedge clk)\n")
    file.write("begin\n")
    file.write("    if (dense_en==1)\n")
    file.write("    begin\n")
    file.write("        re_p=1;\n")
    file.write("        case (marker)\n")           #зависит от размера сверточного блока

    k1=1
    k2=1
    i=2
    STOP=0
    while (STOP==0):
        file.write("            "+str(i)+":begin")
        for j in range(num_conv):
            one="SIZE_"+str(num_conv-j)
            if (num_conv-j-1==0): two="0"
            else: two = "SIZE_"+str(num_conv - j - 1)
            file.write(" p"+str(k1)+str(k2)+" = qp["+one+" - 1:"+two+"];")
            if (k2==9):
                k2=1
                k1+=1
            else: k2+=1

        if (i==0): file.write(" we=0;")
        if (i==1):
            file.write(" if (i!=1) begin go=1; j=j+1; end")
            STOP=1
        if (i==2): file.write(" go=0;")

        if (i==3): file.write(" re_w=1;  if (i!=3) dp="+Ypl[:-1]+"+dp;")
        if (i==num_conv+3): file.write(" re_w=0;")
        if ((i>=3)&(i<3+num_conv)): file.write(" read_addressw="+str(i-3)+"+j*("+Y_use_pl[:-1]+");")
        if ((i>=5)&(i<5+num_conv)): file.write(" w"+str(i-4)+"1=qw[SIZE_9-1:SIZE_8]; w"+str(i-4)+"2=qw[SIZE_8-1:SIZE_7]; w"+str(i-4)+"3=qw[SIZE_7-1:SIZE_6]; w"+str(i-4)+"4=qw[SIZE_6-1:SIZE_5]; w"+str(i-4)+"5=qw[SIZE_5-1:SIZE_4]; w"+str(i-4)+"6=qw[SIZE_4-1:SIZE_3]; w"+str(i-4)+"7=qw[SIZE_3-1:SIZE_2]; w"+str(i-4)+"8=qw[SIZE_2-1:SIZE_1]; w"+str(i-4)+"9=qw[SIZE_1-1:0];")
        file.write(" end\n")
        if (i!=8): i=i+1
        else: i=0

    file.write("            default: $display(\"Check case dense\");\n")
    file.write("        endcase\n\n")
    file.write("        read_addressp=memstartp+i;\n\n")
    file.write("        if (marker!=8) marker=marker+1; else marker=0;\n")
    file.write("        i=i+1;\n")
    file.write("        if ((i>(in>>(num_conv>>1))+4)&&(marker==4))\n")
    file.write("            begin\n")
    file.write("        	    write_addressp=memstartzap+((lvl+1)>>(num_conv>>1))-1;\n")
    file.write("                dp_check=dp[SIZE_1+SIZE_1-2+1:SIZE_1-1];\n")
    file.write("		        if (dp_check>2**(SIZE_1-1)-1)\n")
    file.write("                begin\n")
    file.write("                    $display(\"OVERFLOW in dense!\");\n")
    file.write("                    dp_shift=2**(SIZE_1-1)-1;\n")
    file.write("                end\n")
    file.write("                else dp_shift=dp[SIZE_1+SIZE_1-2:SIZE_1-1];\n")
    file.write("                if ((dp_shift<0)&&(nozero==0)) dp_shift=0;\n")

    file.write("                if (sh==0) res=0;\n")
    for i in range(num_conv):
        one="SIZE_"+str(num_conv-i)
        if ((num_conv-i-1)==0): two="0"
        else: two="SIZE_"+str(num_conv-i-1)
        file.write("                if (sh=="+str(i)+") res["+one+"-1:"+two+"]=dp_shift;\n")

    file.write("                lvl=lvl+1;\n")
    file.write("                i=0; \n")
    file.write("                dp=0; \n")
    file.write("                marker=0;\n")
    file.write("                sh=sh+1; if (sh==num_conv) sh=0; \n")
    file.write("		        if ((sh==0)||(lvl==out)) we=1;\n")
    file.write("                if (lvl==out) begin STOP=1; if (sh!=0) write_addressp=write_addressp+1; end\n")
    file.write("    end\n")
    file.write("end\n")
    file.write("else\n")
    file.write("begin\n")
    file.write("    marker=0;\n")
    file.write("    i=0;\n")
    file.write("    j=0;\n")
    file.write("    sh=0;\n")
    file.write("    we=0;\n")
    file.write("    dp=0;\n")
    file.write("    res=0;\n")
    file.write("    re_p=0;\n")
    file.write("    re_w=0;\n")
    file.write("    STOP=0;\n")
    file.write("    lvl=0;\n")
    file.write("end\n")
    file.write("end\n")

    for i in range(num_conv):
        file.write("assign Y"+str(i+1)+"_use=(in_dense>=9*"+str(i)+"+1)?1'b1:1'b0;\n")

    file.write("endmodule\n")

    file.close()


def conv(directory, razmer):
    file = open(directory + "conv.v", 'w')

    bit_matrix = len(bin(razmer)) - 2
    bit_matrix_2 = len(bin(razmer*razmer)) - 2

    file.write("module conv(clk, Y1, prov, matrix, matrix2, i, w1, w2, w3, w4, w5, w6, w7, w8, w9,w11, w12, w13, w14, w15, w16, w17, w18, w19, conv_en, dense_en);\n")
    file.write("parameter SIZE=23;\n")
    file.write("input clk;\n")
    file.write("output reg signed [SIZE+SIZE-2:0] Y1;\n")
    file.write("input [1:0] prov;\n")
    file.write("input ["+str(bit_matrix-1)+":0] matrix;\n")
    file.write("input ["+str(bit_matrix_2-1)+":0] matrix2;\n")
    file.write("input ["+str(bit_matrix_2-1)+":0] i;\n")
    file.write("input signed [SIZE-1:0] w1, w2, w3, w4, w5, w6, w7, w8, w9;\n")
    file.write("input signed [SIZE-1:0] w11, w12, w13, w14, w15, w16, w17, w18, w19;\n")
    file.write("input conv_en;\n")
    file.write("input dense_en;\n")
    file.write("\n")
    file.write("always @(posedge clk)\n")
    file.write("    begin\n")
    file.write("		if (conv_en==1)\n")
    file.write("			begin\n")
    file.write("        Y1=0;\n")
    file.write("        Y1 = Y1+Y(w1,w11);\n")
    file.write("		  //$display(\"center:\",w1,w11);\n")
    file.write("        //right\n")
    file.write("        if ((prov!=2'b10)||(dense_en==1))\n")
    file.write("            begin\n")
    file.write("                Y1 = Y1+Y(w2,w12);\n")
    file.write("					 //$display(\"right:\",w2,w12);\n")
    file.write("            end\n")
    file.write("        //left\n")
    file.write("        if ((prov!=2'b11)||(dense_en==1))\n")
    file.write("            begin\n")
    file.write("                Y1 = Y1+Y(w3,w13);\n")
    file.write("					 //$display(\"left:\",w3,w13);\n")
    file.write("            end\n")
    file.write("        //downleft\n")
    file.write("        if (((i<matrix2-matrix)&&(prov!=2'b11))||(dense_en==1))\n")
    file.write("            begin\n")
    file.write("                Y1 = Y1+Y(w4,w14);\n")
    file.write("					 //$display(\"downleft:\",w4,w14);\n")
    file.write("            end\n")
    file.write("        //upright\n")
    file.write("        if (((i>matrix-1'b1)&&(prov!=2'b10))||(dense_en==1))\n")
    file.write("            begin\n")
    file.write("                Y1 = Y1+Y(w5,w15);\n")
    file.write("					 //$display(\"upright:\",w5,w15);\n")
    file.write("            end\n")
    file.write("        //down\n")
    file.write("        if ((i<matrix2-matrix)||(dense_en==1))\n")
    file.write("            begin\n")
    file.write("                Y1 = Y1+Y(w6,w16);\n")
    file.write("					 //$display(\"down:\",w6,w16);\n")
    file.write("            end\n")
    file.write("        //up\n")
    file.write("        if ((i>matrix-1'b1)||(dense_en==1))\n")
    file.write("            begin\n")
    file.write("                Y1 = Y1+Y(w7,w17);\n")
    file.write("					 //$display(\"up:\",w7,w17);\n")
    file.write("            end\n")
    file.write("        //downright\n")
    file.write("        if (((i<matrix2-matrix)&&(prov!=2'b10))||(dense_en==1))\n")
    file.write("            begin\n")
    file.write("                Y1 = Y1+Y(w8,w18);\n")
    file.write("					 //$display(\"downright:\",w8,w18);\n")
    file.write("            end\n")
    file.write("        //upleft\n")
    file.write("        if (((i>matrix-1'b1)&&(prov!=2'b11))||(dense_en==1))\n")
    file.write("            begin\n")
    file.write("                Y1 = Y1+Y(w9,w19);\n")
    file.write("					 //$display(\"upleft:\",w9,w19);\n")
    file.write("            end\n")
    file.write("			end\n")
    file.write("    end\n\n")
    file.write("function signed [SIZE+SIZE-2:0] Y;\n")
    file.write("    input signed [SIZE-1:0] a, b;\n")
    file.write("    begin\n")
    file.write("        Y = a*b;\n")
    file.write("        //Y = Y>>SIZE-1;\n")
    file.write("    end\n")
    file.write("endfunction\n")
    file.write("\n")
    file.write("endmodule\n")

    file.close()


def RAMtoMEM(directory, max_address_value, steps_count, in_dense_razmer, conv_block_size, num_conv):
    file = open(directory + "RAMtoMEM.v", 'w')

    bit_max_address_value = len(bin(max_address_value)) - 2
    bit_weight_case = len(bin(conv_block_size*conv_block_size)) - 2
    bit_steps_count = len(bin(steps_count)) - 2
    bit_in_dense_razmer = len(bin(in_dense_razmer)) - 2

    file.write("module memorywork(clk,data,address,we_p,we_w,re_RAM,nextstep,dp,dw,addrp,addrw,step_out,GO,in_dense);\n")
    file.write("\n")
    file.write("parameter num_conv=0;\n")
    file.write("\n")
    file.write("parameter picture_size=0;\n")
    file.write("parameter convolution_size=0;\n")
    file.write("parameter SIZE_1=0;\n")
    file.write("parameter SIZE_2=0;\n")
    file.write("parameter SIZE_3=0;\n")
    file.write("parameter SIZE_4=0;\n")
    file.write("parameter SIZE_5=0;\n")
    file.write("parameter SIZE_6=0;\n")
    file.write("parameter SIZE_7=0;\n")
    file.write("parameter SIZE_8=0;\n")
    file.write("parameter SIZE_9=0;\n")
    file.write("parameter SIZE_address_pix=0;\n")
    file.write("parameter SIZE_address_wei=0;\n")
    file.write("\n")
    file.write("inout clk;\n")
    file.write("input signed [SIZE_1-1:0] data;\n")
    file.write("output ["+str(bit_max_address_value-1)+":0] address;\n")
    file.write("output reg we_p;\n")
    file.write("output reg we_w;\n")
    file.write("inout re_RAM;\n")
    file.write("input nextstep;\n")
    file.write("output reg signed [SIZE_"+str(num_conv)+"-1:0] dp;   //write data\n")
    file.write("output reg signed [SIZE_9-1:0] dw;     //write weight\n")
    file.write("output reg [SIZE_address_pix-1:0] addrp;\n")
    file.write("output reg [SIZE_address_wei-1:0] addrw;\n")
    file.write("output ["+str(bit_steps_count-1)+":0] step_out;\n")
    file.write("input GO;\n")
    file.write("input ["+str(bit_in_dense_razmer-1)+":0] in_dense;\n")
    file.write("		  \n")
    file.write("reg [SIZE_address_pix-1:0] addr;\n")
    file.write("wire ["+str(bit_max_address_value-1)+":0] firstaddr,lastaddr;\n")
    file.write("reg sh;\n")
    file.write("\n")
    file.write("reg ["+str(bit_steps_count-1)+":0] step;\n")
    file.write("reg ["+str(bit_steps_count-1)+":0] step_n;\n")
    file.write("reg ["+str(bit_weight_case)+":0] weight_case;\n")
    file.write("reg [SIZE_9-1:0] buff;\n")
    file.write("reg ["+str(bit_max_address_value-1)+":0] i;\n")
    file.write("reg ["+str(bit_max_address_value-1)+":0] i_d;\n")
    file.write("reg ["+str(bit_max_address_value-1)+":0] i1;\n")
    file.write("addressRAM #(.picture_size(picture_size), .convolution_size(convolution_size)) inst_1(.step(step_out),.re_RAM(re_RAM),.firstaddr(firstaddr),.lastaddr(lastaddr));\n")
    file.write("initial sh=0;\n")
    file.write("initial weight_case=0;\n")
    file.write("initial i=0;\n")
    file.write("initial i_d=0;\n")
    file.write("initial i1=0;\n")
    file.write("always @(posedge clk)\n")
    file.write("    begin\n")
    file.write("		  if (GO==1) step=1;\n")
    file.write("		  sh=sh+1;\n")
    file.write("				if (step_out==1)\n")
    file.write("				   begin\n")
    file.write("					if ((i<=lastaddr-firstaddr)&&(sh==0))\n")
    file.write("                    begin\n")
    file.write("                        //address=firstaddr+i;\n")
    file.write("								addr=i;\n")
    file.write("								if (step_out==1) we_p=1;\n")
    file.write("					end\n")
    file.write("                if ((i<=lastaddr-firstaddr)&&(sh==1))\n")
    file.write("                    begin\n")
    file.write("                        if (we_p) \n")
    file.write("									begin\n")
    file.write("											addrp=addr;\n")
    file.write("											dp=0;\n")
    one="SIZE_"+str(num_conv)
    if (num_conv==1): two="0"
    else: two="SIZE_"+str(num_conv-1)
    file.write("											dp["+one+"-1:"+two+"]=data;\n")
    file.write("											we_p=0;\n")
    file.write("									end\n")
    file.write("									i=i+1;\n")
    file.write("							end\n")
    file.write("					 if ((i>lastaddr-firstaddr)&&(sh==1))\n")
    file.write("                    begin\n")
    file.write("                        step=step+1;          //next step\n")
    file.write("                        i=0;\n")
    file.write("                    end\n")
    file.write("				end\n")
    file.write("				if ((step_out==2)||(step_out==4)||(step_out==6)||(step_out==8)||(step_out==10)||(step_out==12)||(step_out==14))\n")
    file.write("					begin\n")
    file.write("						if ((i<=lastaddr-firstaddr)&&(sh==0))\n")
    file.write("                    begin\n")
    file.write("								addr=i1;\n")
    file.write("						  end\n")
    file.write("						if ((i<=lastaddr-firstaddr)&&(sh==1))\n")
    file.write("                    begin\n")
    file.write("										we_w=0;\n")
    file.write("										addrw=addr;\n")
    file.write("										if (weight_case!=0) i=i+1; \n")
    file.write("										if (step_out==14) if (i_d==(in_dense)) begin  dw=buff; we_w=1; weight_case=1; i_d=0; i1=i1+1; end\n")
    file.write("										case (weight_case)\n")
    file.write("											0: ;\n")
    file.write("											1: begin buff=0; buff[SIZE_9-1:SIZE_8]=data; end   \n")
    file.write("											2: buff[SIZE_8-1:SIZE_7]=data; \n")
    file.write("											3: buff[SIZE_7-1:SIZE_6]=data;  \n")
    file.write("											4: buff[SIZE_6-1:SIZE_5]=data;  \n")
    file.write("											5: buff[SIZE_5-1:SIZE_4]=data;           \n")
    file.write("											6: buff[SIZE_4-1:SIZE_3]=data;  \n")
    file.write("											7: buff[SIZE_3-1:SIZE_2]=data;  \n")
    file.write("											8: buff[SIZE_2-1:SIZE_1]=data;   \n")
    file.write("											9: begin buff[SIZE_1-1:0]=data;  i1=i1+1; end\n")
    file.write("											default: $display(\"Check weight_case\");\n")
    file.write("										endcase\n")
    file.write("										if (weight_case!=0) i_d=i_d+1;\n")
    file.write("										if (weight_case==9) begin weight_case=1; dw=buff; we_w=1; end else weight_case=weight_case+1;\n")
    file.write("										\n")
    file.write("                    end\n")
    file.write("						if ((i>lastaddr-firstaddr)&&(sh==1))\n")
    file.write("                    begin\n")
    file.write("                        step=step+1;          //next step\n")
    file.write("                        i=0;\n")
    file.write("								i_d=0;\n")
    file.write("								i1=0;\n")
    file.write("								weight_case=0;\n")
    file.write("                    end\n")
    file.write("            end\n")
    file.write("				else\n")
    file.write("					we_w=0;\n")
    file.write("    end\n")
    file.write("always @(posedge nextstep) if (GO==1) step_n=0; else step_n=step_n+1;\n")
    file.write("assign step_out=step+step_n;\n")
    file.write("assign address=firstaddr+i;\n")
    file.write("endmodule\n")
    file.close()


def addressRAM(directory, steps_count, max_address_value):
    file = open(directory + "addressRAM.v", 'w')

    bit_steps_count = len(bin(steps_count)) - 2
    bit_max_address_value = len(bin(max_address_value)) - 2

    file.write("module addressRAM(\n")
    file.write("	input ["+str(bit_steps_count-1)+":0] step,\n")
    file.write("	output reg re_RAM,\n")
    file.write("	output reg ["+str(bit_max_address_value-1)+":0] firstaddr, lastaddr\n")
    file.write(");\n")
    file.write("parameter picture_size = 0;\n")
    file.write("parameter convolution_size = 0;\n")
    file.write("\n")
    file.write("parameter picture_storage_limit = picture_size*picture_size;\n")
    file.write("parameter convweight = picture_storage_limit + (1*4 + 4*4 + 4*8 + 8*8) * convolution_size;  // all convolution weights [784:1828]\n")
    file.write("\n")
    file.write("parameter conv1 = picture_storage_limit + 1*4 * convolution_size;\n")
    file.write("parameter conv2 = picture_storage_limit + (1*4 + 4*4) * convolution_size;\n")
    file.write("parameter conv3 = picture_storage_limit + (1*4 + 4*4 + 4*8) * convolution_size;\n")
    file.write("parameter conv4 = picture_storage_limit + (1*4 + 4*4 + 4*8 + 8*8) * convolution_size;\n")
    file.write("parameter conv5 = picture_storage_limit + (1*4 + 4*4 + 4*8 + 8*8 + 8*16) * convolution_size;\n")
    file.write("parameter conv6 = picture_storage_limit + (1*4 + 4*4 + 4*8 + 8*8 + 8*16 + 16*16) * convolution_size;\n")
    file.write("\n")
    file.write("parameter dense = conv6+176;\n")
    file.write("\n")
    file.write("always @(step)\n")
    file.write("case (step)\n")
    file.write("1'd1: begin       //picture\n")
    file.write("		firstaddr = 0;\n")
    file.write("		lastaddr = picture_storage_limit;\n")
    file.write("		re_RAM = 1;\n")
    file.write("	  end \n")
    file.write("2'd2: begin       //weights conv1 \n")
    file.write("		firstaddr = picture_storage_limit;\n")
    file.write("		lastaddr = conv1;\n")
    file.write("		re_RAM = 1;\n")
    file.write("	  end\n")
    file.write("3'd4: begin			//weights conv2\n")
    file.write("		firstaddr = conv1;\n")
    file.write("      lastaddr = conv2;\n")
    file.write("		re_RAM = 1;\n")
    file.write("      end		\n")
    file.write("3'd6: begin			//weights conv3\n")
    file.write("		firstaddr = conv2;\n")
    file.write("		lastaddr = conv3;\n")
    file.write("		re_RAM = 1;\n")
    file.write("	  end\n")
    file.write("4'd8: begin			//weights conv4\n")
    file.write("		firstaddr = conv3;\n")
    file.write("		lastaddr = conv4;\n")
    file.write("		re_RAM = 1;\n")
    file.write("		end\n")
    file.write("4'd10: begin		//weights conv5\n")
    file.write("		firstaddr = conv4;\n")
    file.write("		lastaddr = conv5;\n")
    file.write("		re_RAM = 1;\n")
    file.write("	  end\n")
    file.write("4'd12: begin		//weights conv6\n")
    file.write("		firstaddr = conv5;\n")
    file.write("		lastaddr = conv6;\n")
    file.write("		re_RAM = 1;\n")
    file.write("	  end\n")
    file.write("4'd14: begin		//weights conv7\n")
    file.write("		firstaddr = conv6;\n")
    file.write("		lastaddr =  dense;\n")
    file.write("		re_RAM = 1;\n")
    file.write("	  end\n")
    file.write("default:\n")
    file.write("			begin\n")
    file.write("				re_RAM = 0;\n")
    file.write("			end\n")
    file.write("endcase\n")
    file.write("endmodule\n")

    file.close()


def database(directory, max_address_value, razmer, size, list):
    file = open(directory + "database.v", 'w')

    bit_max_address_value = len(bin(max_address_value)) - 2

    file.write("module database(clk,datata,re,address,we,dp,address_p);\n")
    file.write("\n")
    file.write("parameter SIZE=0;\n")
    file.write("\n")
    file.write("input clk;\n")
    file.write("output reg signed [SIZE-1:0] datata;\n")
    file.write("input re,we;\n")
    file.write("input ["+str(bit_max_address_value-1)+":0] address;\n")
    file.write("input signed [SIZE-1:0] dp;\n")
    file.write("input ["+str(bit_max_address_value-1)+":0] address_p;\n")
    file.write("\n")
    file.write("reg signed [SIZE-1:0] storage ["+str(max_address_value-1)+":0];\n")
    file.write("\n")
    file.write("initial begin\n")
    file.write("\n")

    for i in range(razmer*razmer):
        file.write("storage["+str(i)+"] =  "+str(size)+"'b0;\n")
    for i in range(len(list)):
        if (list[i]>=0): minus=" "
        else: minus="-"
        data = bin(abs(list[i]))[2:]
        while (len(data)<size):
            data = str(0) + data
        file.write("storage["+str(i+razmer*razmer)+"] = "+minus+str(size)+"'b"+str(data)+"; // "+str(list[i])+"\n")

    file.write("end\n")
    file.write("\n")
    file.write("always @(posedge clk) if (we==1) storage[address_p] <= dp;\n")
    file.write("always @(posedge clk) if (re==1) datata<=storage[address];\n")
    file.write("\n")
    file.write("endmodule\n")

    file.close()


def conv_TOP(directory, razmer, max_conv_input_size, num_conv):
    file = open(directory + "conv_TOP.v", 'w')

    bit_matrix=len(bin(razmer))-2
    bit_matrix_2=len(bin(razmer*razmer))-2
    bit_max_conv_input_size = len(bin(max_conv_input_size)) - 2
    Y=''
    w=''
    res_out=''
    res=''
    res_old=''
    globmaxp_perem=''
    res_bias_check=''
    for i in range(num_conv):
        Y = Y + "Y" + str(i + 1) + ","
        w = w + "w"+str(i+1)+"5,w"+str(i+1)+"4,w"+str(i+1)+"6,w"+str(i+1)+"3,w"+str(i+1)+"7,w"+str(i+1)+"2,w"+str(i+1)+"8,w"+str(i+1)+"1,w"+str(i+1)+"9,"
        res_out = res_out + "res_out_" +str(i+1) + ","
        res = res + "res"+ str(i+1) + ","
        res_old = res_old + "res_old_" + str(i+1) + ","
        globmaxp_perem = globmaxp_perem + "globmaxp_perem_" + str(i+1) + ","
        res_bias_check = res_bias_check + "res_bias_check_" + str(i+1) + ","

    file.write("module conv_TOP(clk,conv_en,STOP,memstartp,memstartw,memstartzap,read_addressp,write_addressp,read_addresstp,write_addresstp,read_addressw,we,re_wb,re,we_t,re_t,qp,qtp,qw,dp,dtp,prov,matrix,matrix2,i_2,lvl,slvl,"+Y+w+"p1,p2,p3,p8,p7,p4,p5,p9,p6,go,num,filt,bias,globmaxp_en);\n")
    file.write("\n")
    file.write("parameter num_conv;\n")
    file.write("parameter SIZE_1=0;\n")
    file.write("parameter SIZE_2=0;\n")
    file.write("parameter SIZE_3=0;\n")
    file.write("parameter SIZE_4=0;\n")
    file.write("parameter SIZE_5=0;\n")
    file.write("parameter SIZE_6=0;\n")
    file.write("parameter SIZE_7=0;\n")
    file.write("parameter SIZE_8=0;\n")
    file.write("parameter SIZE_9=0;\n")
    file.write("parameter SIZE_address_pix=0;\n")
    file.write("parameter SIZE_address_pix_t=0;\n")
    file.write("parameter SIZE_address_wei=0;\n")
    file.write("\n")
    file.write("input clk,conv_en,globmaxp_en;\n")
    file.write("input [1:0] prov;\n")
    file.write("input ["+str(bit_matrix-1)+":0] matrix;\n")
    file.write("input ["+str(bit_matrix_2-1)+":0] matrix2;\n")
    file.write("input [SIZE_address_pix-1:0] memstartp; \n")
    file.write("input [SIZE_address_wei-1:0] memstartw;\n")
    file.write("input [SIZE_address_pix-1:0] memstartzap;                  																	\n")
    file.write("input ["+str(bit_max_conv_input_size-1)+":0] lvl;\n")
    file.write("input [1:0] slvl;\n")
    file.write("output reg [SIZE_address_pix-1:0] read_addressp;\n")
    file.write("output reg [SIZE_address_pix_t-1:0] read_addresstp;\n")
    file.write("output reg [SIZE_address_wei-1:0] read_addressw;\n")
    file.write("output reg [SIZE_address_pix-1:0] write_addressp;\n")
    file.write("output reg [SIZE_address_pix_t-1:0] write_addresstp;\n")
    file.write("output reg we,re,re_wb;\n")
    file.write("output reg we_t,re_t;\n")
    file.write("input signed [SIZE_"+str(num_conv)+"-1:0] qp;\n")
    file.write("input signed [SIZE_2*"+str(num_conv)+"-1:0] qtp;\n")
    file.write("input signed [SIZE_9-1:0] qw;\n")
    file.write("output signed [SIZE_"+str(num_conv)+"-1:0] dp;\n")
    file.write("output signed [SIZE_2*"+str(num_conv)+"-1:0] dtp;\n")
    file.write("output reg STOP;\n")
    file.write("output ["+str(bit_matrix_2-1)+":0] i_2;\n")
    file.write("input signed [SIZE_1+SIZE_1-2:0] "+Y[:-1]+";\n")
    file.write("output reg signed [SIZE_1-1:0] "+w[:-1]+";\n")
    file.write("output reg signed [SIZE_1-1:0]p1,p2,p3,p4,p5,p6,p7,p8,p9;\n")
    file.write("output reg go;\n")
    file.write("input [2:0] num;\n")
    file.write("input ["+str(bit_max_conv_input_size-1)+":0] filt;\n")
    file.write("input bias;\n")
    file.write("\n")
    file.write("reg signed [SIZE_1-1:0] "+res_out[:-1]+";\n")
    file.write("reg signed [SIZE_1+SIZE_1-2+1:0] "+res[:-1]+";\n")
    file.write("reg signed [SIZE_1+SIZE_1-2+1:0] "+res_old[:-1]+";\n")
    file.write("reg signed [SIZE_1-1:0] "+globmaxp_perem[:-1]+";\n")
    file.write("\n")
    file.write("reg signed [SIZE_1-1:0] buff0 [2:0];\n")
    file.write("reg signed [SIZE_1-1:0] buff1 [2:0];\n")
    file.write("reg signed [SIZE_1-1:0] buff2 [2:0];\n")
    file.write("\n")
    file.write("reg [3:0] marker;\n")
    file.write("reg zagryzka_weight;\n")
    file.write("reg ["+str(bit_matrix_2-1)+":0] i;\n")
    file.write("\n")
    file.write("reg signed [SIZE_1-1+1:0] "+res_bias_check[:-1]+";\n")
    file.write("\n")
    file.write("initial zagryzka_weight=0;\n")
    file.write("initial marker=0;\n")
    file.write("\n")
    file.write("always @(posedge clk)\n")
    file.write("begin\n")
    file.write("if (conv_en==1)        //enable convolution\n")
    file.write("	begin\n")
    file.write("		if (zagryzka_weight==0)        \n")
    file.write("		begin\n")
    file.write("		   case (marker)\n")

    for i in range(num_conv+3):
        file.write("				"+str(i)+": begin")
        if (i==0): file.write(" re_wb=1;")
        if (i<num_conv): file.write(" read_addressw=memstartw+(2'd"+str(i)+"*(filt+1));")
        if ((i<num_conv+2)&(i>=2)):
            file.write("							w"+str(i-1)+"1=qw[SIZE_1-1:0]; \n")
            file.write("							w"+str(i-1)+"2=qw[SIZE_2-1:SIZE_1]; \n")
            file.write("							w"+str(i-1)+"3=qw[SIZE_3-1:SIZE_2]; \n")
            file.write("							w"+str(i-1)+"4=qw[SIZE_4-1:SIZE_3]; \n")
            file.write("							w"+str(i-1)+"5=qw[SIZE_5-1:SIZE_4]; \n")
            file.write("							w"+str(i-1)+"6=qw[SIZE_6-1:SIZE_5]; \n")
            file.write("							w"+str(i-1)+"7=qw[SIZE_7-1:SIZE_6]; \n")
            file.write("							w"+str(i-1)+"8=qw[SIZE_8-1:SIZE_7]; \n")
            file.write("							w"+str(i-1)+"9=qw[SIZE_9-1:SIZE_8]; \n")
        if (i==num_conv+2): file.write(" zagryzka_weight=1; re_wb=0; marker=-1;")
        file.write(" end\n")

    file.write("				default: $display(\"Check zagryzka_weight\");\n")
    file.write("		endcase\n")
    file.write("		marker=marker+1;\n")
    file.write("		end\n")
    file.write("		else\n")
    file.write("		begin\n")
    file.write("			re=1;\n")
    file.write("			case (marker)\n")
    file.write("				0: begin		\n")
    file.write("								re_t=0;\n")
    file.write("								read_addressp=i+memstartp; \n")
    file.write("								if ((i-1)<matrix2-matrix) \n")
    file.write("								begin\n")

    lvl=''
    for i in range(len(bin(num_conv))-3):
        lvl = ",lvl["+str(i)+"]" + lvl
    for i in range(num_conv):
        one_size="SIZE_"+str(num_conv-i)
        if ((num_conv-i-1)==0): two_size="0"
        else: two_size="SIZE_"+str(num_conv-i-1)
        if (num_conv>1):
            if (i==0):
                file.write("								if")
            else:
                file.write("								else if")
            file.write(" ({"+lvl[1:]+"}=="+str(len(bin(num_conv))-3)+"'d"+str(i)+") buff2[2]=qp["+one_size+"-1:"+two_size+"];\n")
        else: file.write("								buff2[2]=qp[SIZE_1-1:0];\n")

    file.write("								end\n")
    file.write("								else buff2[2]=0;\n")
    file.write("								\n")
    file.write("								if (i>=2) go=1;\n")
    file.write("								\n")
    file.write("								\n")
    file.write("								p1=buff1[1];  //center\n")
    file.write("								p2=buff1[2];  //right\n")
    file.write("								p3=buff1[0];  //left\n")
    file.write("								p8=buff2[0];  //downright\n")
    file.write("								p7=buff0[2];  //up\n")
    file.write("								p4=buff2[1];  //downleft \n")
    file.write("								p5=buff0[1];  //upright\n")
    file.write("								p9=buff2[2];  //upleft\n")
    file.write("								p6=buff0[0];  //down \n")
    file.write("								\n")
    file.write("								\n")
    file.write("					end\n")
    file.write("				1: begin		if (i>=matrix-1) read_addressp=i-matrix+memstartp;\n")

    for i in range(num_conv):
        one_size="(SIZE_2)*"+str(num_conv-i)
        if ((num_conv-i-1)==0): two_size="0"
        else: two_size="(SIZE_2)*"+str(num_conv-i-1)
        file.write("								res_old_"+str(i+1)+"=qtp["+one_size+"-1:"+two_size+"];\n")

    file.write("								\n")
    file.write("								go=0;\n")
    file.write("								\n")
    file.write("								buff2[0]=buff2[1];\n")
    file.write("								buff1[0]=buff1[1];\n")
    file.write("								buff0[0]=buff0[1];\n")
    file.write("								buff2[1]=buff2[2];\n")
    file.write("								buff1[1]=buff1[2];\n")
    file.write("								buff0[1]=buff0[2];\n")
    file.write("					end\n")
    file.write("				2: begin    if (i<matrix2-matrix) read_addressp=i+matrix+memstartp;\n")

    lvl=''
    for i in range(len(bin(num_conv))-3):
        lvl = ",lvl["+str(i)+"]" + lvl
    for i in range(num_conv):
        one_size="SIZE_"+str(num_conv-i)
        if ((num_conv-i-1)==0): two_size="0"
        else: two_size="SIZE_"+str(num_conv-i-1)
        if (num_conv>1):
            if (i==0):
                file.write("								if")
            else:
                file.write("								else if")
            file.write(" ({"+lvl[1:]+"}=="+str(len(bin(num_conv))-3)+"'d"+str(i)+") buff1[2]=qp["+one_size+"-1:"+two_size+"];\n")
        else: file.write("								buff1[2]=qp[SIZE_1-1:0];\n")

    file.write("								\n")
    file.write("								if (i>=2) \n")
    file.write("								begin\n")
    file.write("								we_t=1;\n")
    file.write("								write_addresstp=i-2+matrix2*num+(slvl*((filt+1)*matrix2)>>(num_conv>>1));\n")
    file.write("								if (globmaxp_en)  write_addressp=memstartzap;\n")
    file.write("								else	write_addressp=memstartzap+i-2;\n")

    for i in range(num_conv):
        file.write("								res"+str(i+1)+"=Y"+str(i+1)+"; if (lvl!=0) res"+str(i+1)+"=res"+str(i+1)+"+res_old_"+str(i+1)+"; \n")

    file.write("								if (bias==1) \n")
    file.write("									begin  \n")

    for i in range(num_conv):
        file.write("										res_bias_check_"+str(i+1)+"=res"+str(i+1)+"[SIZE_1+SIZE_1-2+1:SIZE_1-1];\n")
        file.write("										if (res_bias_check_"+str(i+1)+">(2**(SIZE_1-1))-1) \n")
        file.write("											begin\n")
        file.write("												$display(\"OVERFLOW in conv!\");\n")
        file.write("												res_out_"+str(i+1)+"=(2**(SIZE_1-1))-1;\n")
        file.write("											end\n")
        file.write("										else res_out_"+str(i+1)+"=res"+str(i+1)+"[SIZE_1+SIZE_1-2:SIZE_1-1];\n")
        file.write("										if (res_out_"+str(i+1)+"<0) res_out_"+str(i+1)+"=0; \n")

    file.write("										\n")
    file.write("										if (globmaxp_en)\n")
    file.write("											begin\n")

    for i in range(num_conv):
        file.write("												if (res_out_"+str(i+1)+">globmaxp_perem_"+str(i+1)+") globmaxp_perem_"+str(i+1)+"=res_out_"+str(i+1)+";\n")

    file.write("										   end\n")
    file.write("										we=1;\n")
    file.write("									end\n")
    file.write("								end\n")
    file.write("					end\n")
    file.write("				3: begin		\n")
    file.write("								re_t=1;\n")
    file.write("								read_addresstp=i-1+matrix2*num+slvl*(((filt+1)*matrix2>>(num_conv>>1)));\n")
    file.write("								if (i>=matrix-1)\n")
    file.write("								begin\n")

    lvl = ''
    for i in range(len(bin(num_conv)) - 3):
        lvl = ",lvl[" + str(i) + "]" + lvl
    for i in range(num_conv):
        one_size = "SIZE_" + str(num_conv - i)
        if ((num_conv - i - 1) == 0):
            two_size = "0"
        else:
            two_size = "SIZE_" + str(num_conv - i - 1)
        if (num_conv > 1):
            if (i == 0):
                file.write("								if")
            else:
                file.write("								else if")
            file.write(" ({" + lvl[1:] + "}==" + str(len(bin(num_conv)) - 3) + "'d" + str(i) + ") buff0[2]=qp[" + one_size + "-1:" + two_size + "];\n")
        else:
            file.write("								buff0[2]=qp[SIZE_1-1:0];\n")

    file.write("								end\n")
    file.write("								else buff0[2]=0;\n")
    file.write("								\n")
    file.write("								we_t=0;\n")
    file.write("								we=0;\n")
    file.write("					end						\n")
    file.write("			default: $display(\"Check case conv_TOP\");\n")
    file.write("			endcase\n")
    file.write("			\n")
    file.write("			if (marker!=3) marker=marker+1; \n")
    file.write("			else begin \n")
    file.write("					marker=0; \n")
    file.write("					if (i<matrix2+1) i=i+1; \n")
    file.write("					else STOP=1; \n")
    file.write("				  end\n")
    file.write("		end\n")
    file.write("	end\n")
    file.write("else \n")
    file.write("	begin\n")
    file.write("		i=0;\n")
    file.write("		zagryzka_weight=0;\n")
    file.write("		STOP=0;\n")
    file.write("		re=0;\n")
    file.write("		re_t=0;\n")
    file.write("		go=0;\n")
    file.write("		marker=0;\n")

    for i in range(num_conv):
        file.write("		globmaxp_perem_"+str(i+1)+"=0;\n")

    file.write("	end\n")
    file.write("end\n")
    file.write("assign i_2=i-2;\n")

    globmaxp_perem=''
    res_out=''
    res=''
    for i in range(num_conv):
        globmaxp_perem=globmaxp_perem+"globmaxp_perem_"+str(i+1)+","
        res_out=res_out+"res_out_"+str(i+1)+","
        res=res+"res"+str(i+1)+","

    file.write("assign dp=(globmaxp_en)?{"+globmaxp_perem[:-1]+"}:{"+res_out[:-1]+"};\n")
    file.write("assign dtp={"+res[:-1]+"};\n")
    file.write("endmodule\n")

    file.close()


def result(directory,output_neurons_count,num_conv):
    file = open(directory + "result.v", 'w')

    bit_output_neurons_count = len(bin(output_neurons_count))-2
    bit_marker_chislo = len(bin(output_neurons_count+2)) - 2

    file.write("module result(clk,enable,STOP,memstartp,read_addressp,qp,re,RESULT);\n")
    file.write("\n")
    file.write("parameter SIZE_1=0;\n")
    file.write("parameter SIZE_2=0;\n")
    file.write("parameter SIZE_3=0;\n")
    file.write("parameter SIZE_4=0;\n")
    file.write("parameter SIZE_address_pix=0;\n")
    file.write("\n")
    file.write("input clk,enable;\n")
    file.write("output reg STOP;\n")
    file.write("input [SIZE_address_pix-1:0] memstartp;\n")
    file.write("input [SIZE_"+str(num_conv)+"-1:0] qp;\n")
    file.write("output reg re;\n")
    file.write("output reg [SIZE_address_pix-1:0] read_addressp;\n")
    file.write("output reg ["+str(bit_output_neurons_count-1)+":0] RESULT;\n")
    file.write("\n")
    file.write("reg ["+str(bit_marker_chislo-1)+":0] marker;\n")
    file.write("reg signed [SIZE_1-1:0] buff;\n")
    file.write("\n")

    p=''
    for i in range(num_conv):
        p=p+" p"+str(i+1)+","

    file.write("wire signed [SIZE_1-1:0] "+p[:-1]+";\n")
    file.write("always @(posedge clk)\n")
    file.write("begin\n")
    file.write("if (enable==1)\n")
    file.write("begin\n")
    file.write("re=1;\n")
    file.write("case (marker)\n")

    RESULT=0
    stop=0
    i=0
    while (stop==0):
        file.write("	"+str(i)+": begin")
        if (stop==0): file.write(" read_addressp=memstartp+"+str(i)+";")
        if (i==1): file.write(" buff = 0;")
        if (i>=2):
            for j in range(num_conv):
                file.write(" if (p"+str(j+1)+">=buff) begin buff=p"+str(j+1)+"; RESULT="+str(RESULT)+"; end")
                RESULT+=1
                if (RESULT==11):
                    stop=1
                    break
            if (stop==1): file.write(" STOP=1;")
        file.write(" end\n")
        i+=1


    file.write("	default: $display(\"Check case result\");\n")
    file.write("endcase\n")
    file.write("marker=marker+1;\n")
    file.write("end\n")
    file.write("else \n")
    file.write("begin\n")
    file.write("re=0;\n")
    file.write("marker=0;\n")
    file.write("STOP=0;\n")
    file.write("end\n")
    file.write("end\n")
    file.write("\n")

    for i in range(num_conv):
        one="SIZE_"+str(num_conv-i)
        if (num_conv-i-1==0): two="0"
        else: two="SIZE_"+str(num_conv-i-1)
        file.write("assign p"+str(i+1)+"=qp["+one+"-1:"+two+"];\n")

    file.write("endmodule\n")

    file.close()


def TOP(directory, size, razmer, max_address_value, output_neurons_count, max_weights_per_layer,
        total_conv_layers_number, total_maxp_layers_number, max_conv_input_size, in_dense_razmer,
        out_dense_razmer, max_conv_output_size, layers, num_conv):
    file = open(directory + "TOP.v", 'w')

    bit_max_address_value = len(bin(max_address_value)) - 2
    bit_output_neurons_count = len(bin(output_neurons_count)) - 2
    bit_address_pix = len(bin(razmer*razmer*8+razmer*razmer)) - 2
    bit_address_pix_t = len(bin(razmer*razmer*4)) - 2
    bit_max_weights_per_layer = len(bin(max_weights_per_layer)) - 2
    bit_total_conv_layers_number = len(bin(total_conv_layers_number)) - 2
    bit_total_maxp_layers_number = len(bin(total_maxp_layers_number)) - 2
    bit_TOPlvl_chislo = len(bin(total_conv_layers_number+total_maxp_layers_number)) - 2
    bit_max_conv_input_size = len(bin(max_conv_input_size)) - 2
    bit_razmer = len(bin(razmer)) - 2
    bit_razmer_2 = len(bin(razmer*razmer)) - 2
    bit_in_dense_razmer = len(bin(in_dense_razmer)) - 2
    bit_out_dense_razmer = len(bin(out_dense_razmer)) - 2
    bit_max_conv_output_size = len(bin(max_conv_output_size)) - 2
    Y=''
    p=''
    w=''
    p_d=''
    w_c=''
    w_d=''
    for i in range(num_conv):
        Y=Y+"Y"+str(i+1)+","
        w_c=w_c+"w"+str(i+1)+"1_c,w"+str(i+1)+"2_c,w"+str(i+1)+"3_c,w"+str(i+1)+"4_c,w"+str(i+1)+"5_c,w"+str(i+1)+"6_c,w"+str(i+1)+"7_c,w"+str(i+1)+"8_c,w"+str(i+1)+"9_c,"
        w_d=w_d+"w"+str(i+1)+"1_d,w"+str(i+1)+"2_d,w"+str(i+1)+"3_d,w"+str(i+1)+"4_d,w"+str(i+1)+"5_d,w"+str(i+1)+"6_d,w"+str(i+1)+"7_d,w"+str(i+1)+"8_d,w"+str(i+1)+"9_d,"
        p_d=p_d+"p"+str(i+1)+"1_d,p"+str(i+1)+"2_d,p"+str(i+1)+"3_d,p"+str(i+1)+"4_d,p"+str(i+1)+"5_d,p"+str(i+1)+"6_d,p"+str(i+1)+"7_d,p"+str(i+1)+"8_d,p"+str(i+1)+"9_d,"
        p = p + "p" + str(i + 1) + "1,p" + str(i + 1) + "2,p" + str(i + 1) + "3,p" + str(i + 1) + "4,p" + str(i + 1) + "5,p" + str(i + 1) + "6,p" + str(i + 1) + "7,p" + str(i + 1) + "8,p" + str(i + 1) + "9,"
        w = w + "w" + str(i + 1) + "1,w" + str(i + 1) + "2,w" + str(i + 1) + "3,w" + str(i + 1) + "4,w" + str(i + 1) + "5,w" + str(i + 1) + "6,w" + str(i + 1) + "7,w" + str(i + 1) + "8,w" + str(i + 1) + "9,"

    file.write("module TOP(clk, GO, RESULT, we_database, dp_database, address_p_database, STOP);\n")
    file.write("\n")
    file.write("parameter num_conv = "+str(num_conv)+";\n")
    file.write("parameter SIZE_1 = "+str(size)+";\n")
    file.write("parameter SIZE_2 = SIZE_1*2;\n")
    file.write("parameter SIZE_3 = SIZE_1*3;\n")
    file.write("parameter SIZE_4 = SIZE_1*4;\n")
    file.write("parameter SIZE_5 = SIZE_1*5;\n")
    file.write("parameter SIZE_6 = SIZE_1*6;\n")
    file.write("parameter SIZE_7 = SIZE_1*7;\n")
    file.write("parameter SIZE_8 = SIZE_1*8;\n")
    file.write("parameter SIZE_9 = SIZE_1*9;\n")
    file.write("parameter SIZE_address_pix = "+str(bit_address_pix)+";\n")
    file.write("parameter SIZE_address_pix_t = "+str(bit_address_pix_t)+";\n")
    file.write("parameter SIZE_address_wei = "+str(bit_max_weights_per_layer)+";\n")
    file.write("parameter picture_size = "+str(razmer)+";\n")
    file.write("parameter picture_storage_limit = 0;\n")
    file.write("parameter razmpar = picture_size >> 1;\n")
    file.write("parameter razmpar2  = picture_size >> 2;\n")
    file.write("parameter picture_storage_limit_2 = ((picture_size*picture_size)*4) >> (num_conv >> 1);\n")
    file.write("parameter convolution_size = 9;\n")
    file.write("input clk;\n")
    file.write("input GO;\n")
    file.write("output ["+str(bit_output_neurons_count-1)+":0] RESULT;\n")
    file.write("input we_database;\n")
    file.write("input signed [SIZE_1-1:0] dp_database;\n")
    file.write("input ["+str(bit_max_address_value-1)+":0] address_p_database;\n")
    file.write("output reg STOP;\n")
    file.write("\n")
    file.write("wire signed [SIZE_1-1:0] data;\n")
    file.write("wire re_RAM;\n")
    file.write("wire ["+str(bit_max_address_value-1)+":0] address;\n")
    file.write("\n")
    file.write("reg conv_en;\n")
    file.write("wire STOP_conv;\n")
    file.write("\n")
    file.write("reg maxp_en;\n")
    file.write("wire STOP_maxp;\n")
    file.write("\n")
    file.write("reg dense_en;\n")
    file.write("wire STOP_dense;	 \n")
    file.write("\n")
    file.write("reg result_en;\n")
    file.write("wire STOP_res;	\n")
    file.write("wire ["+str(bit_output_neurons_count-1)+":0] res_out;\n")
    file.write("\n")
    file.write("reg bias,globmaxp_en;\n")
    file.write("	 \n")
    file.write("reg ["+str(bit_total_conv_layers_number-1)+":0] TOPlvl_conv;\n")
    file.write("reg ["+str(bit_total_maxp_layers_number-1)+":0] TOPlvl_maxp;\n")
    file.write("wire ["+str(bit_TOPlvl_chislo-1)+":0] TOPlvl;\n")
    file.write("reg ["+str(bit_max_conv_input_size-1)+":0] lvl;\n")
    file.write("reg [1:0] slvl;\n")
    file.write("reg [2:0] num;\n")
    file.write("reg [2:0] num_maxp;\n")
    file.write("reg [SIZE_address_pix-1:0] memstartp;\n")
    file.write("wire [SIZE_address_pix-1:0] memstartp_lvl;\n")
    file.write("reg [SIZE_address_wei-1:0] memstartw;\n")
    file.write("wire [SIZE_address_wei-1:0] memstartw_lvl;\n")
    file.write("reg [SIZE_address_pix-1:0] memstartzap;\n")
    file.write("wire [SIZE_address_pix-1:0] memstartzap_num;\n")
    file.write("wire [SIZE_address_pix-1:0] read_addressp;\n")
    file.write("wire [SIZE_address_pix_t-1:0] read_addresstp;\n")
    file.write("wire [SIZE_address_wei-1:0] read_addressw;\n")
    file.write("wire [SIZE_address_pix-1:0] read_addressp_conv;\n")
    file.write("wire [SIZE_address_pix-1:0] read_addressp_maxp;\n")
    file.write("wire [SIZE_address_pix-1:0] read_addressp_dense;\n")
    file.write("wire [SIZE_address_pix-1:0] read_addressp_res;\n")
    file.write("wire [SIZE_address_wei-1:0] read_addressw_conv;\n")
    file.write("wire [SIZE_address_wei-1:0] read_addressw_dense;\n")
    file.write("wire [SIZE_address_pix-1:0] write_addressp;\n")
    file.write("wire [SIZE_address_pix_t-1:0] write_addresstp;\n")
    file.write("wire [SIZE_address_wei-1:0] write_addressw;\n")
    file.write("wire [SIZE_address_pix-1:0] write_addressp_zagr;\n")
    file.write("wire [SIZE_address_pix-1:0] write_addressp_conv;\n")
    file.write("wire [SIZE_address_pix-1:0] write_addressp_maxp;\n")
    file.write("wire [SIZE_address_pix-1:0] write_addressp_dense;\n")
    file.write("wire we_p,we_tp,we_w;\n")
    file.write("wire re_p,re_tp,re_w;\n")
    file.write("wire we_p_zagr;\n")
    file.write("wire we_conv,re_wb_conv,re_conv;\n")
    file.write("wire we_maxp,re_maxp;\n")
    file.write("wire we_dense,re_p_dense,re_w_dense;\n")
    file.write("wire re_p_res;\n")
    file.write("wire signed [SIZE_"+str(num_conv)+"-1:0] qp;\n")
    file.write("wire signed [(SIZE_2)*"+str(num_conv)+"-1:0] qtp;\n")
    file.write("wire signed [SIZE_9-1:0] qw;\n")
    file.write("wire signed [SIZE_"+str(num_conv)+"-1:0] dp;\n")
    file.write("wire signed [(SIZE_2)*"+str(num_conv)+"-1:0] dtp;\n")
    file.write("wire signed [SIZE_9-1:0] dw;\n")
    file.write("wire signed [SIZE_"+str(num_conv)+"-1:0] dp_conv;\n")
    file.write("wire signed [SIZE_"+str(num_conv)+"-1:0] dp_maxp;\n")
    file.write("wire signed [SIZE_"+str(num_conv)+"-1:0] dp_dense;\n")
    file.write("wire signed [SIZE_"+str(num_conv)+"-1:0] dp_zagr;\n")
    file.write("\n")
    file.write("wire [1:0] prov;\n")
    file.write("wire ["+str(bit_razmer_2-1)+":0] i_conv;\n")
    file.write("wire signed [SIZE_2-2:0] "+Y[:-1]+";\n")
    file.write("wire signed [SIZE_1-1:0] "+w[:-1]+";\n")
    file.write("wire signed [SIZE_1-1:0] "+p[:-1]+";\n")
    file.write("wire signed [SIZE_1-1:0] "+w_c[:-1]+";\n")
    file.write("wire signed [SIZE_1-1:0] p1_c,p2_c,p3_c,p4_c,p5_c,p6_c,p7_c,p8_c,p9_c;\n")
    file.write("wire signed [SIZE_1-1:0] "+w_d[:-1]+";\n")
    file.write("wire signed [SIZE_1-1:0] "+p_d[:-1]+";\n")
    file.write("wire go_conv;\n")
    file.write("wire go_conv_TOP;\n")
    file.write("wire go_dense;\n")
    file.write("\n")
    file.write("wire [4:0] step;\n")
    file.write("reg nextstep;\n")
    file.write("\n")
    file.write("reg ["+str(bit_razmer-1)+":0] matrix;\n")
    file.write("wire ["+str(bit_razmer_2-1)+":0] matrix2;\n")
    file.write("\n")
    file.write("reg ["+str(bit_max_conv_output_size-1)+":0] mem;\n")
    file.write("reg ["+str(bit_max_conv_input_size-1)+":0] filt;\n")
    file.write("\n")
    file.write("reg ["+str(bit_in_dense_razmer-1)+":0] in_dense;\n")
    file.write("reg ["+str(bit_out_dense_razmer-1)+":0] out_dense;\n")
    file.write("reg nozero_dense;\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")
    file.write("database #(SIZE_1) database (.clk(clk),.datata(data),.re(re_RAM),.address(address),.we(we_database),.dp(dp_database),.address_p(address_p_database));\n")
    file.write("conv_TOP #(num_conv,SIZE_1,SIZE_2,SIZE_3,SIZE_4,SIZE_5,SIZE_6,SIZE_7,SIZE_8,SIZE_9,SIZE_address_pix,SIZE_address_pix_t,SIZE_address_wei) conv(clk,conv_en,STOP_conv,memstartp_lvl,memstartw_lvl,memstartzap_num,read_addressp_conv,write_addressp_conv,read_addresstp,write_addresstp,read_addressw_conv,we_conv,re_wb_conv,re_conv,we_tp,re_tp,qp,qtp,qw,dp_conv,dtp,prov,matrix,matrix2,i_conv,lvl,slvl,"+Y+w_c+"p1_c,p2_c,p3_c,p4_c,p5_c,p6_c,p7_c,p8_c,p9_c,go_conv_TOP,num,filt,bias,globmaxp_en);\n")
    file.write("memorywork #(num_conv,picture_size,convolution_size,SIZE_1,SIZE_2,SIZE_3,SIZE_4,SIZE_5,SIZE_6,SIZE_7,SIZE_8,SIZE_9,SIZE_address_pix,SIZE_address_wei) block(.clk(clk),.we_p(we_p_zagr),.we_w(we_w),.re_RAM(re_RAM),.addrp(write_addressp_zagr),.addrw(write_addressw),.dp(dp_zagr),.dw(dw),.step_out(step),.nextstep(nextstep),.data(data),.address(address),.GO(GO),.in_dense(in_dense));\n")
    file.write("RAM #(picture_size,SIZE_1,SIZE_2,SIZE_4,SIZE_9,SIZE_address_pix,SIZE_address_pix_t,SIZE_address_wei) memory(qp,qtp,qw,dp,dtp,dw,write_addressp,read_addressp,write_addresstp,read_addresstp,write_addressw,read_addressw,we_p,we_tp,we_w,re_p,re_tp,re_w,clk);\n")
    file.write("border border(clk,conv_en,i_conv,matrix,prov);\n")
    file.write("maxp #(SIZE_1,SIZE_2,SIZE_3,SIZE_4,SIZE_address_pix) maxpooling(clk,maxp_en,memstartp_lvl,memstartzap_num,read_addressp_maxp,write_addressp_maxp,re_maxp,we_maxp,qp,dp_maxp,STOP_maxp,matrix2,matrix);\n")
    file.write("dense #(num_conv,SIZE_1,SIZE_2,SIZE_3,SIZE_4,SIZE_5,SIZE_6,SIZE_7,SIZE_8,SIZE_9,SIZE_address_pix,SIZE_address_wei) dense(clk,dense_en,STOP_dense,in_dense,out_dense,we_dense,re_p_dense,re_w_dense,read_addressp_dense,read_addressw_dense,write_addressp_dense,memstartp_lvl,memstartzap_num,qp,qw,dp_dense,"+Y+w_d+p_d+"go_dense,nozero_dense,in_dense);\n")
    file.write("result #(SIZE_1,SIZE_2,SIZE_3,SIZE_4,SIZE_address_pix) result(clk,result_en,STOP_res,memstartp_lvl,read_addressp_res,qp,re_p_res,res_out);\n")
    file.write("\n")
    for i in range(num_conv):
        file.write("conv #(SIZE_1) conv"+str(i+1)+" (clk,Y"+str(i+1)+",prov,matrix,matrix2,i_conv,p"+str(i+1)+"1,p"+str(i+1)+"2,p"+str(i+1)+"3,p"+str(i+1)+"4,p"+str(i+1)+"5,p"+str(i+1)+"6,p"+str(i+1)+"7,p"+str(i+1)+"8,p"+str(i+1)+"9,w"+str(i+1)+"1,w"+str(i+1)+"2,w"+str(i+1)+"3,w"+str(i+1)+"4,w"+str(i+1)+"5,w"+str(i+1)+"6,w"+str(i+1)+"7,w"+str(i+1)+"8,w"+str(i+1)+"9,go_conv,dense_en);\n")
    file.write("\n")
    file.write("\n")
    file.write("initial lvl = 0;\n")
    file.write("initial slvl = 0;\n")
    file.write("initial num = 0;\n")
    file.write("initial num_maxp = 0;\n")
    file.write("initial memstartw = 0;\n")
    file.write("")
    file.write("\n")
    file.write("always @(posedge clk )\n")
    file.write("begin\n")
    file.write("    if (GO==1)\n")
    file.write("    begin\n")
    file.write("        STOP=0;\n")
    file.write("        nextstep=1;\n")
    file.write("        num_maxp=0;\n")
    file.write("        globmaxp_en=0;\n")
    file.write("        TOPlvl_maxp=0;\n")
    file.write("        matrix=picture_size;\n")
    file.write("        dense_en=0;\n")
    file.write("    end\n")
    file.write("    else nextstep=0;\n")
    file.write("\n")
    file.write("    if (STOP==0)\n")
    file.write("    begin\n")

    TOPlvl = 1
    step = 1
    one = "picture_storage_limit"
    two = "picture_storage_limit_2"
    start = 1
    for i in range(len(layers)):
        layer = layers[i]
        if 'Conv2D' in str(type(layer)):
            mem = layer.output_shape[3]-1
            filt = layer.input_shape[3]-1

            if (start == 0):  file.write("	    if ((TOPlvl=="+str(TOPlvl)+")&&(step=="+str(step)+")) nextstep = 1;\n")
            else: start = 0

            step += 2

            file.write("	    if ((TOPlvl=="+str(TOPlvl)+")&&(step=="+str(step)+"))\n")
            file.write("		    begin\n")
            file.write("			    memstartp = "+str(one)+";\n")
            file.write("			    memstartzap = "+str(two)+";\n")
            file.write("			    conv_en = 1;\n")
            file.write("			    mem = "+str(mem)+";\n")
            file.write("			    filt = "+str(filt)+";\n")
            file.write("			    matrix = "+str(layer.input_shape[1])+";\n")
            if 'GlobalMaxPooling2D' in str(type(layers[i+2])):
                file.write("			    globmaxp_en = 1;\n")
            else:   file.write("			    globmaxp_en = 0;\n")
            file.write("		    end	\n")

            one_t=one
            two_t=two
            two=one_t
            one=two_t
            TOPlvl += 1
        elif 'MaxPooling2D' in str(type(layer)):
            if not 'GlobalMaxPooling2D' in str(type(layer)):
                for i in range(int(layer.input_shape[3]/4)):
                    file.write("	    if ((TOPlvl=="+str(TOPlvl)+")&&(STOP_maxp==0))\n")
                    file.write("		    begin\n")
                    file.write("			    memstartp = "+str(one)+"+"+str(i)+"*matrix2*((4 >> (num_conv >> 1)));\n")
                    file.write("			    memstartzap = "+str(two)+"+"+str(i)+"*(matrix2 >> (num_conv >> 1));\n")
                    file.write("			    maxp_en = 1;\n")
                    file.write("		    end\n")

                    TOPlvl += 1

                one_t = one
                two_t = two
                two = one_t
                one = two_t
        elif 'Dense' in str(type(layer)):
            file.write("	    if ((TOPlvl=="+str(TOPlvl)+")&&(step=="+str(step)+")) \n")
            file.write("            begin \n")
            file.write("                globmaxp_en = 0; \n")
            file.write("                nextstep = 1; \n")
            file.write("                in_dense = "+str(layer.input_shape[1])+"; \n")
            file.write("                out_dense = "+str(layer.output_shape[1])+"; \n")
            file.write("            end   \n")

            step += 2

            file.write("	    if ((TOPlvl=="+str(TOPlvl)+")&&(STOP_dense==0)&&(step=="+str(step)+"))\n")
            file.write("		    begin\n")
            file.write("			    memstartp = "+str(one)+";\n")
            file.write("			    memstartzap = "+str(two)+";\n")
            file.write("			    dense_en = 1;\n")
            if ((i + 2) >= len(layers)):
                file.write("    			nozero_dense = 1;\n")
            else:
                file.write("    			nozero_dense = 0;\n")
            file.write("	    	end\n")

            one_t = one
            two_t = two
            two = one_t
            one = two_t
        if (i == len(layers)-1):
            step += 1
            file.write("	    if ((TOPlvl=="+str(TOPlvl)+")&&(STOP_dense==0)&&(step=="+str(step)+"))\n")
            file.write("		    begin\n")
            file.write("			    memstartp = "+str(one)+";\n")
            file.write("		    	result_en = 1;\n")
            file.write("		    end\n")

    file.write("	    if (lvl==filt) bias = 1;\n")
    file.write("        else bias = 0;\n")
    file.write("	    if ((STOP_conv)&&(conv_en==1)) conv_en = 0;\n")
    file.write("	    if ((STOP_maxp==1)&&(maxp_en==1)) \n")
    file.write("            begin \n")
    file.write("                maxp_en=0; \n")
    file.write("                if (num_maxp!=4-num_conv) num_maxp=num_maxp+1; \n")
    file.write("                else \n")
    file.write("                    begin \n")
    file.write("                        num_maxp=0; \n")
    file.write("                        TOPlvl_maxp=TOPlvl_maxp+1; \n")
    file.write("                    end  \n")
    file.write("            end\n")
    file.write("	    if (STOP_dense==1) \n")
    file.write("            begin \n")
    file.write("                dense_en = 0; \n")
    file.write("                nextstep = 1; \n")
    file.write("            end\n")
    file.write("	    if ((STOP_res==1)&&(result_en==1))\n")
    file.write("	    begin\n")
    file.write("	    	result_en = 0;\n")
    file.write("	    	STOP = 1;\n")
    file.write("    	end\n")
    file.write("    end\n")
    file.write("end\n")
    file.write("\n")
    file.write("always @(negedge STOP_conv or posedge GO)\n")
    file.write("	begin\n")
    file.write("		if (GO)\n")
    file.write("			begin\n")
    file.write("				lvl = 0;\n")
    file.write("				slvl = 0;\n")
    file.write("				TOPlvl_conv = 1;\n")
    file.write("			end\n")
    file.write("		else\n")
    file.write("			begin\n")
    file.write("				if (num==0)\n")
    file.write("					begin \n")
    file.write("						if (mem!=(4+(slvl*4))-1) slvl=slvl+1; \n")
    file.write("						else \n")
    file.write("                            begin \n")
    file.write("                                slvl=0; \n")
    file.write("                                lvl=lvl+1; \n")
    file.write("                            end \n")
    file.write("					end\n")
    file.write("				if (lvl==(filt+1))  \n")
    file.write("					begin\n")
    file.write("						lvl = 0;\n")
    file.write("						TOPlvl_conv = TOPlvl_conv+1'b1;\n")
    file.write("					end\n")
    file.write("			end\n")
    file.write("	end\n")
    file.write("\n")
    file.write("assign memstartw_lvl = memstartw+lvl+slvl*(4*(filt+1))+num*(filt+1)*num_conv;\n")
    file.write("assign memstartzap_num = memstartzap+(((globmaxp_en==1)&&(lvl==filt))?(slvl*(4>>(num_conv>>1))+num):((conv_en==1)?(num*matrix2+slvl*matrix2*((4>>(num_conv>>1)))):((maxp_en==1)?num_maxp*(matrix2>>2):0)));\n")
    file.write("assign memstartp_lvl = memstartp+(lvl>>(num_conv>>1))*matrix2+((maxp_en==1)?num_maxp*matrix2:0);   //new!\n")
    file.write("	\n")
    file.write("assign re_p = (conv_en==1)?re_conv:((maxp_en==1)?re_maxp:((dense_en==1)?re_p_dense:((result_en==1)?re_p_res:0)));\n")
    file.write("assign re_w = (conv_en==1)?re_wb_conv:((dense_en==1)?re_w_dense:0);\n")
    file.write("assign read_addressp = (conv_en==1)?read_addressp_conv:((maxp_en==1)?read_addressp_maxp:((dense_en==1)?read_addressp_dense:((result_en==1)?read_addressp_res:0)));\n")
    file.write("assign we_p = (step==1)?we_p_zagr:((conv_en==1)?we_conv:((maxp_en==1)?we_maxp:((dense_en==1)?we_dense:0)));\n")
    file.write("assign dp = (step==1)?dp_zagr:((conv_en==1)?dp_conv:((maxp_en==1)?dp_maxp:((dense_en==1)?dp_dense:0)));\n")
    file.write("assign write_addressp = (step==1)?write_addressp_zagr:((conv_en==1)?write_addressp_conv:((maxp_en==1)?write_addressp_maxp:((dense_en==1)?write_addressp_dense:0)));\n")
    file.write("assign read_addressw = (conv_en==1)?read_addressw_conv:((dense_en==1)?read_addressw_dense:0);\n")
    file.write("\n")
    file.write("assign matrix2 = matrix*matrix;\n")
    file.write("\n")
    for i in range(num_conv):
        file.write("assign p"+str(i+1)+"1 = (conv_en==1)?p1_c:((dense_en==1)?p"+str(i+1)+"1_d:0);  //center\n")
        file.write("assign p"+str(i+1)+"2 = (conv_en==1)?p2_c:((dense_en==1)?p"+str(i+1)+"2_d:0);  //right\n")
        file.write("assign p"+str(i+1)+"3 = (conv_en==1)?p3_c:((dense_en==1)?p"+str(i+1)+"3_d:0);  //left\n")
        file.write("assign p"+str(i+1)+"4 = (conv_en==1)?p4_c:((dense_en==1)?p"+str(i+1)+"4_d:0);  //downleft\n")
        file.write("assign p"+str(i+1)+"5 = (conv_en==1)?p5_c:((dense_en==1)?p"+str(i+1)+"5_d:0);  //upright\n")
        file.write("assign p"+str(i+1)+"6 = (conv_en==1)?p6_c:((dense_en==1)?p"+str(i+1)+"6_d:0);  //down\n")
        file.write("assign p"+str(i+1)+"7 = (conv_en==1)?p7_c:((dense_en==1)?p"+str(i+1)+"7_d:0);  //up\n")
        file.write("assign p"+str(i+1)+"8 = (conv_en==1)?p8_c:((dense_en==1)?p"+str(i+1)+"8_d:0);  //downright\n")
        file.write("assign p"+str(i+1)+"9 = (conv_en==1)?p9_c:((dense_en==1)?p"+str(i+1)+"9_d:0);  //upleft\n")
    file.write("\n")
    for i in range(num_conv):
        file.write("assign w"+str(i+1)+"1 = (conv_en==1)?w"+str(i+1)+"1_c:((dense_en==1)?w"+str(i+1)+"1_d:0);\n")
        file.write("assign w"+str(i+1)+"2 = (conv_en==1)?w"+str(i+1)+"2_c:((dense_en==1)?w"+str(i+1)+"2_d:0);\n")
        file.write("assign w"+str(i+1)+"3 = (conv_en==1)?w"+str(i+1)+"3_c:((dense_en==1)?w"+str(i+1)+"3_d:0);\n")
        file.write("assign w"+str(i+1)+"4 = (conv_en==1)?w"+str(i+1)+"4_c:((dense_en==1)?w"+str(i+1)+"4_d:0);\n")
        file.write("assign w"+str(i+1)+"5 = (conv_en==1)?w"+str(i+1)+"5_c:((dense_en==1)?w"+str(i+1)+"5_d:0);\n")
        file.write("assign w"+str(i+1)+"6 = (conv_en==1)?w"+str(i+1)+"6_c:((dense_en==1)?w"+str(i+1)+"6_d:0);\n")
        file.write("assign w"+str(i+1)+"7 = (conv_en==1)?w"+str(i+1)+"7_c:((dense_en==1)?w"+str(i+1)+"7_d:0);\n")
        file.write("assign w"+str(i+1)+"8 = (conv_en==1)?w"+str(i+1)+"8_c:((dense_en==1)?w"+str(i+1)+"8_d:0);\n")
        file.write("assign w"+str(i+1)+"9 = (conv_en==1)?w"+str(i+1)+"9_c:((dense_en==1)?w"+str(i+1)+"9_d:0);\n")
    file.write("\n")
    file.write("assign TOPlvl = TOPlvl_conv+TOPlvl_maxp;\n")
    file.write("\n")
    file.write("\n")
    file.write("assign go_conv = (conv_en==1)?go_conv_TOP:((dense_en==1)?go_dense:0);\n")
    file.write("\n")
    file.write("assign RESULT = (STOP)?res_out:4'b1111;\n")
    file.write("\n")
    file.write("initial num = 0;\n")
    file.write("always @(posedge STOP_conv) \n")
    file.write("    begin \n")
    file.write("        if (num!=(4 >> (num_conv >> 1))-1) num = num+1; \n")
    file.write("        else num = 0; \n")
    file.write("    end\n")
    file.write("\n")
    file.write("endmodule\n")

    file.close()


def go_mat_model(model, bit_precizion):

    weights = dict()
    list = []

    for i in range(len(model.layers)):
        layer = model.layers[i]

        # Convolution layers
        if 'Conv2D' in str(type(layer)):

            # Standard float convolution
            weights[i] = layer.get_weights()[0]

            # Коэффициент для приведения весов к виду [1;1]
            koeff_weights = 1.0
            weights_converted = preprocess_forward(weights[i].copy(), koeff_weights)
            wgt_bit = convert_to_fix_point(weights_converted.copy(), bit_precizion)
            for i3 in range(len(wgt_bit[0][0][0])):
                for i2 in range(len(wgt_bit[0][0])):
                    for i0 in range(len(wgt_bit)):
                        for i1 in range(len(wgt_bit[0])):
                            list.append(wgt_bit[i0][i1][i2][i3])

        elif 'Dense' in str(type(layer)):

            weights[i] = layer.get_weights()[0]
            weights_converted = weights[i].copy()
            wgt_bit = convert_to_fix_point(weights_converted.copy(), bit_precizion)
            for i1 in range(len(wgt_bit[0])):
                for i0 in range(len(wgt_bit)):
                    list.append(wgt_bit[i0][i1])
    return list


def preprocess_forward(arr, val):
    arr1 = arr.copy().astype(np.float32)
    arr1 /= val
    return arr1


def convert_to_fix_point(arr1, bit):
    arr2 = arr1.copy().astype(np.float32)
    arr2[arr2 < 0] = 0.0
    arr2 = np.round(np.abs(arr2) * (2 ** bit))
    arr3 = arr1.copy().astype(np.float32)
    arr3[arr3 > 0] = 0.0
    arr3 = -np.round(np.abs(-arr3) * (2 ** bit))
    arr4 = arr2 + arr3
    return arr4.astype(np.int64)


if __name__ == '__main__':

    # Where to store neural net verilog
    output_directory = "./verilog/code/neuroset/"
    # Bit size of weights (including sign)
    bit_size = get_optimal_bit_for_weights() + 1
    # Number of convolution blocks (1, 2 or 4). Higher faster, but requires more logic cells.
    num_conv = 1

    print('Create verilog in directory: {}'.format(output_directory))
    print('Bit size: {}'.format(bit_size))
    print('Number of convolution blocks: {}'.format(num_conv))

    print('Read model...')
    model = keras_model_low_weights_digit_detector()
    model.load_weights('weights/keras_model_low_weights_digit_detector_rescaled.h5')

    list = go_mat_model(model, bit_size-1)

    # Maximum size of image for neural net. Equal to 28.
    max_input_image_size = 0
    # Count of data in database: initial image and all weights
    max_address_value = 0
    # Number of steps in neural net. Step means any action like loading data, processing convolution or maxpooling layer.
    steps_count = 0
    # Size of convolution block. Equal to 3.
    conv_block_size = 0
    # Maximum weights for all layers, divided by 9, because all weights packed by 9 numbers, to be called in one tact.
    max_weights_per_layer = 0
    # Maximum number of output feature maps for all convolution layers
    max_conv_output_size = 0
    # Maximum number of input feature maps for all convolution layers
    max_conv_input_size = 0
    # Number of neurons on final classification layer. Equal to 11.
    output_neurons_count = 0
    # Number of convolution layers
    total_conv_layers_number = 0
    # Number of maxpooling layers
    total_maxp_layers_number = 0
    # Number of dense (FC) layers
    total_dense_layers_number = 0

    conv_inputs = []
    conv_mem = []
    conv_filt = []
    dense_inputs = []
    dense_outputs = []
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if 'Input' in str(type(layer)):
            input = layer.input_shape[1]*layer.input_shape[2]
        elif 'Conv2D' in str(type(layer)):
            total_conv_layers_number += 1
            conv_inputs.append(layer.input_shape[1])
            conv_mem.append(layer.input_shape[3])
            conv_filt.append(layer.output_shape[3])
            w = layer.get_weights()
            conv_block_size_1 = len(w[0])
            conv_block_size_2 = len(w[0][0])
            max_address_value += len(w[0][0][0][0])*len(w[0][0][0])*len(w[0][0])*len(w[0])
            max_weights_per_layer_1 = len(w[0][0][0][0])*len(w[0][0][0])
            if max_weights_per_layer_1 > max_weights_per_layer:
                max_weights_per_layer = max_weights_per_layer_1
        elif 'MaxPooling2D' in str(type(layer)):
            if not 'GlobalMaxPooling2D' in str(type(layer)):
                total_maxp_layers_number += int(layer.input_shape[3]/4)
        elif 'Dense' in str(type(layer)):
            w = layer.get_weights()
            total_dense_layers_number += 1
            dense_inputs.append(layer.input_shape[1])
            dense_outputs.append(layer.output_shape[1])
            max_address_value += len(layer.get_weights()[0][0]) * len(w[0])
            max_weights_per_layer_1 = int(len(w[0][0]) * len(w[0])/(conv_block_size_1*conv_block_size_2)) + 1
            if max_weights_per_layer_1 > max_weights_per_layer:
                max_weights_per_layer = max_weights_per_layer_1
        if i == len(model.layers) - 1:
            output_neurons_count = layer.output_shape[1]

    max_input_image_size = max(conv_inputs)
    steps_count = 2 + (total_conv_layers_number*2) + (total_dense_layers_number*2) + 1
    in_dense_size = max(dense_inputs)
    out_dense_size = max(dense_outputs)
    conv_block_size = conv_block_size_1
    max_conv_output_size = max(conv_mem)
    max_conv_input_size = max(conv_filt)
    max_address_value += input

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    print("Make border file")
    border(output_directory, max_input_image_size)
    print("Make maxpooling file")
    maxpooling(output_directory, max_input_image_size, num_conv)
    print("Make RAM file")
    RAM(output_directory, max_weights_per_layer, num_conv)
    print("Make dense file")
    dense(output_directory, in_dense_size, out_dense_size, num_conv)
    print("Make conv file")
    conv(output_directory, max_input_image_size)
    print("Make RAMtoMEM file")
    RAMtoMEM(output_directory, max_address_value, steps_count, in_dense_size, conv_block_size, num_conv)
    print("Make addressRAM file")
    addressRAM(output_directory, steps_count, max_address_value)
    print("Make database file")
    database(output_directory, max_address_value, max_input_image_size, bit_size, list)
    print("Make conv_TOP file")
    conv_TOP(output_directory, max_input_image_size, max_conv_input_size,num_conv)
    print("Make result file")
    result(output_directory, output_neurons_count, num_conv)
    print("Make TOP file")
    TOP(output_directory, bit_size, max_input_image_size, max_address_value, output_neurons_count,
        max_weights_per_layer, total_conv_layers_number, total_maxp_layers_number, max_conv_input_size,
        in_dense_size, out_dense_size, max_conv_output_size, model.layers, num_conv)
