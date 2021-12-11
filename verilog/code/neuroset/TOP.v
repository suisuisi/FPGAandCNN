module TOP(clk, GO, RESULT, we_database, dp_database, address_p_database, STOP);

parameter num_conv = 1;
parameter SIZE_1 = 11;
parameter SIZE_2 = SIZE_1*2;
parameter SIZE_3 = SIZE_1*3;
parameter SIZE_4 = SIZE_1*4;
parameter SIZE_5 = SIZE_1*5;
parameter SIZE_6 = SIZE_1*6;
parameter SIZE_7 = SIZE_1*7;
parameter SIZE_8 = SIZE_1*8;
parameter SIZE_9 = SIZE_1*9;
parameter SIZE_address_pix = 13;
parameter SIZE_address_pix_t = 12;
parameter SIZE_address_wei = 9;
parameter picture_size = 28;
parameter picture_storage_limit = 0;
parameter razmpar = picture_size >> 1;
parameter razmpar2  = picture_size >> 2;
parameter picture_storage_limit_2 = ((picture_size*picture_size)*4) >> (num_conv >> 1);
parameter convolution_size = 9;
input clk;
input GO;
output [3:0] RESULT;
input we_database;
input signed [SIZE_1-1:0] dp_database;
input [12:0] address_p_database;
output reg STOP;

wire signed [SIZE_1-1:0] data;
wire re_RAM;
wire [12:0] address;

reg conv_en;
wire STOP_conv;

reg maxp_en;
wire STOP_maxp;

reg dense_en;
wire STOP_dense;	 

reg result_en;
wire STOP_res;	
wire [3:0] res_out;

reg bias,globmaxp_en;
	 
reg [2:0] TOPlvl_conv;
reg [1:0] TOPlvl_maxp;
wire [3:0] TOPlvl;
reg [4:0] lvl;
reg [1:0] slvl;
reg [2:0] num;
reg [2:0] num_maxp;
reg [SIZE_address_pix-1:0] memstartp;
wire [SIZE_address_pix-1:0] memstartp_lvl;
reg [SIZE_address_wei-1:0] memstartw;
wire [SIZE_address_wei-1:0] memstartw_lvl;
reg [SIZE_address_pix-1:0] memstartzap;
wire [SIZE_address_pix-1:0] memstartzap_num;
wire [SIZE_address_pix-1:0] read_addressp;
wire [SIZE_address_pix_t-1:0] read_addresstp;
wire [SIZE_address_wei-1:0] read_addressw;
wire [SIZE_address_pix-1:0] read_addressp_conv;
wire [SIZE_address_pix-1:0] read_addressp_maxp;
wire [SIZE_address_pix-1:0] read_addressp_dense;
wire [SIZE_address_pix-1:0] read_addressp_res;
wire [SIZE_address_wei-1:0] read_addressw_conv;
wire [SIZE_address_wei-1:0] read_addressw_dense;
wire [SIZE_address_pix-1:0] write_addressp;
wire [SIZE_address_pix_t-1:0] write_addresstp;
wire [SIZE_address_wei-1:0] write_addressw;
wire [SIZE_address_pix-1:0] write_addressp_zagr;
wire [SIZE_address_pix-1:0] write_addressp_conv;
wire [SIZE_address_pix-1:0] write_addressp_maxp;
wire [SIZE_address_pix-1:0] write_addressp_dense;
wire we_p,we_tp,we_w;
wire re_p,re_tp,re_w;
wire we_p_zagr;
wire we_conv,re_wb_conv,re_conv;
wire we_maxp,re_maxp;
wire we_dense,re_p_dense,re_w_dense;
wire re_p_res;
wire signed [SIZE_1-1:0] qp;
wire signed [(SIZE_2)*1-1:0] qtp;
wire signed [SIZE_9-1:0] qw;
wire signed [SIZE_1-1:0] dp;
wire signed [(SIZE_2)*1-1:0] dtp;
wire signed [SIZE_9-1:0] dw;
wire signed [SIZE_1-1:0] dp_conv;
wire signed [SIZE_1-1:0] dp_maxp;
wire signed [SIZE_1-1:0] dp_dense;
wire signed [SIZE_1-1:0] dp_zagr;

wire [1:0] prov;
wire [9:0] i_conv;
wire signed [SIZE_2-2:0] Y1;
wire signed [SIZE_1-1:0] w11,w12,w13,w14,w15,w16,w17,w18,w19;
wire signed [SIZE_1-1:0] p11,p12,p13,p14,p15,p16,p17,p18,p19;
wire signed [SIZE_1-1:0] w11_c,w12_c,w13_c,w14_c,w15_c,w16_c,w17_c,w18_c,w19_c;
wire signed [SIZE_1-1:0] p1_c,p2_c,p3_c,p4_c,p5_c,p6_c,p7_c,p8_c,p9_c;
wire signed [SIZE_1-1:0] w11_d,w12_d,w13_d,w14_d,w15_d,w16_d,w17_d,w18_d,w19_d;
wire signed [SIZE_1-1:0] p11_d,p12_d,p13_d,p14_d,p15_d,p16_d,p17_d,p18_d,p19_d;
wire go_conv;
wire go_conv_TOP;
wire go_dense;

wire [4:0] step;
reg nextstep;

reg [4:0] matrix;
wire [9:0] matrix2;

reg [4:0] mem;
reg [4:0] filt;

reg [4:0] in_dense;
reg [3:0] out_dense;
reg nozero_dense;



database #(SIZE_1) database (.clk(clk),.datata(data),.re(re_RAM),.address(address),.we(we_database),.dp(dp_database),.address_p(address_p_database));
conv_TOP #(num_conv,SIZE_1,SIZE_2,SIZE_3,SIZE_4,SIZE_5,SIZE_6,SIZE_7,SIZE_8,SIZE_9,SIZE_address_pix,SIZE_address_pix_t,SIZE_address_wei) conv(clk,conv_en,STOP_conv,memstartp_lvl,memstartw_lvl,memstartzap_num,read_addressp_conv,write_addressp_conv,read_addresstp,write_addresstp,read_addressw_conv,we_conv,re_wb_conv,re_conv,we_tp,re_tp,qp,qtp,qw,dp_conv,dtp,prov,matrix,matrix2,i_conv,lvl,slvl,Y1,w11_c,w12_c,w13_c,w14_c,w15_c,w16_c,w17_c,w18_c,w19_c,p1_c,p2_c,p3_c,p4_c,p5_c,p6_c,p7_c,p8_c,p9_c,go_conv_TOP,num,filt,bias,globmaxp_en);
memorywork #(num_conv,picture_size,convolution_size,SIZE_1,SIZE_2,SIZE_3,SIZE_4,SIZE_5,SIZE_6,SIZE_7,SIZE_8,SIZE_9,SIZE_address_pix,SIZE_address_wei) block(.clk(clk),.we_p(we_p_zagr),.we_w(we_w),.re_RAM(re_RAM),.addrp(write_addressp_zagr),.addrw(write_addressw),.dp(dp_zagr),.dw(dw),.step_out(step),.nextstep(nextstep),.data(data),.address(address),.GO(GO),.in_dense(in_dense));
RAM #(picture_size,SIZE_1,SIZE_2,SIZE_4,SIZE_9,SIZE_address_pix,SIZE_address_pix_t,SIZE_address_wei) memory(qp,qtp,qw,dp,dtp,dw,write_addressp,read_addressp,write_addresstp,read_addresstp,write_addressw,read_addressw,we_p,we_tp,we_w,re_p,re_tp,re_w,clk);
border border(clk,conv_en,i_conv,matrix,prov);
maxp #(SIZE_1,SIZE_2,SIZE_3,SIZE_4,SIZE_address_pix) maxpooling(clk,maxp_en,memstartp_lvl,memstartzap_num,read_addressp_maxp,write_addressp_maxp,re_maxp,we_maxp,qp,dp_maxp,STOP_maxp,matrix2,matrix);
dense #(num_conv,SIZE_1,SIZE_2,SIZE_3,SIZE_4,SIZE_5,SIZE_6,SIZE_7,SIZE_8,SIZE_9,SIZE_address_pix,SIZE_address_wei) dense(clk,dense_en,STOP_dense,in_dense,out_dense,we_dense,re_p_dense,re_w_dense,read_addressp_dense,read_addressw_dense,write_addressp_dense,memstartp_lvl,memstartzap_num,qp,qw,dp_dense,Y1,w11_d,w12_d,w13_d,w14_d,w15_d,w16_d,w17_d,w18_d,w19_d,p11_d,p12_d,p13_d,p14_d,p15_d,p16_d,p17_d,p18_d,p19_d,go_dense,nozero_dense,in_dense);
result #(SIZE_1,SIZE_2,SIZE_3,SIZE_4,SIZE_address_pix) result(clk,result_en,STOP_res,memstartp_lvl,read_addressp_res,qp,re_p_res,res_out);

conv #(SIZE_1) conv1 (clk,Y1,prov,matrix,matrix2,i_conv,p11,p12,p13,p14,p15,p16,p17,p18,p19,w11,w12,w13,w14,w15,w16,w17,w18,w19,go_conv,dense_en);


initial lvl = 0;
initial slvl = 0;
initial num = 0;
initial num_maxp = 0;
initial memstartw = 0;

always @(posedge clk )
begin
    if (GO==1)
    begin
        STOP=0;
        nextstep=1;
        num_maxp=0;
        globmaxp_en=0;
        TOPlvl_maxp=0;
        matrix=picture_size;
        dense_en=0;
    end
    else nextstep=0;

    if (STOP==0)
    begin
	    if ((TOPlvl==1)&&(step==3))
		    begin
			    memstartp = picture_storage_limit;
			    memstartzap = picture_storage_limit_2;
			    conv_en = 1;
			    mem = 3;
			    filt = 0;
			    matrix = 28;
			    globmaxp_en = 0;
		    end	
	    if ((TOPlvl==2)&&(step==3)) nextstep = 1;
	    if ((TOPlvl==2)&&(step==5))
		    begin
			    memstartp = picture_storage_limit_2;
			    memstartzap = picture_storage_limit;
			    conv_en = 1;
			    mem = 3;
			    filt = 3;
			    matrix = 28;
			    globmaxp_en = 0;
		    end	
	    if ((TOPlvl==3)&&(STOP_maxp==0))
		    begin
			    memstartp = picture_storage_limit+0*matrix2*((4 >> (num_conv >> 1)));
			    memstartzap = picture_storage_limit_2+0*(matrix2 >> (num_conv >> 1));
			    maxp_en = 1;
		    end
	    if ((TOPlvl==4)&&(step==5)) nextstep = 1;
	    if ((TOPlvl==4)&&(step==7))
		    begin
			    memstartp = picture_storage_limit_2;
			    memstartzap = picture_storage_limit;
			    conv_en = 1;
			    mem = 7;
			    filt = 3;
			    matrix = 14;
			    globmaxp_en = 0;
		    end	
	    if ((TOPlvl==5)&&(step==7)) nextstep = 1;
	    if ((TOPlvl==5)&&(step==9))
		    begin
			    memstartp = picture_storage_limit;
			    memstartzap = picture_storage_limit_2;
			    conv_en = 1;
			    mem = 7;
			    filt = 7;
			    matrix = 14;
			    globmaxp_en = 0;
		    end	
	    if ((TOPlvl==6)&&(STOP_maxp==0))
		    begin
			    memstartp = picture_storage_limit_2+0*matrix2*((4 >> (num_conv >> 1)));
			    memstartzap = picture_storage_limit+0*(matrix2 >> (num_conv >> 1));
			    maxp_en = 1;
		    end
	    if ((TOPlvl==7)&&(STOP_maxp==0))
		    begin
			    memstartp = picture_storage_limit_2+1*matrix2*((4 >> (num_conv >> 1)));
			    memstartzap = picture_storage_limit+1*(matrix2 >> (num_conv >> 1));
			    maxp_en = 1;
		    end
	    if ((TOPlvl==8)&&(step==9)) nextstep = 1;
	    if ((TOPlvl==8)&&(step==11))
		    begin
			    memstartp = picture_storage_limit;
			    memstartzap = picture_storage_limit_2;
			    conv_en = 1;
			    mem = 15;
			    filt = 7;
			    matrix = 7;
			    globmaxp_en = 0;
		    end	
	    if ((TOPlvl==9)&&(step==11)) nextstep = 1;
	    if ((TOPlvl==9)&&(step==13))
		    begin
			    memstartp = picture_storage_limit_2;
			    memstartzap = picture_storage_limit;
			    conv_en = 1;
			    mem = 15;
			    filt = 15;
			    matrix = 7;
			    globmaxp_en = 1;
		    end	
	    if ((TOPlvl==10)&&(step==13)) 
            begin 
                globmaxp_en = 0; 
                nextstep = 1; 
                in_dense = 16; 
                out_dense = 11; 
            end   
	    if ((TOPlvl==10)&&(STOP_dense==0)&&(step==15))
		    begin
			    memstartp = picture_storage_limit;
			    memstartzap = picture_storage_limit_2;
			    dense_en = 1;
    			nozero_dense = 1;
	    	end
	    if ((TOPlvl==10)&&(STOP_dense==0)&&(step==16))
		    begin
			    memstartp = picture_storage_limit_2;
		    	result_en = 1;
		    end
	    if (lvl==filt) bias = 1;
        else bias = 0;
	    if ((STOP_conv)&&(conv_en==1)) conv_en = 0;
	    if ((STOP_maxp==1)&&(maxp_en==1)) 
            begin 
                maxp_en=0; 
                if (num_maxp!=4-num_conv) num_maxp=num_maxp+1; 
                else 
                    begin 
                        num_maxp=0; 
                        TOPlvl_maxp=TOPlvl_maxp+1; 
                    end  
            end
	    if (STOP_dense==1) 
            begin 
                dense_en = 0; 
                nextstep = 1; 
            end
	    if ((STOP_res==1)&&(result_en==1))
	    begin
	    	result_en = 0;
	    	STOP = 1;
    	end
    end
end

always @(negedge STOP_conv or posedge GO)
	begin
		if (GO)
			begin
				lvl = 0;
				slvl = 0;
				TOPlvl_conv = 1;
			end
		else
			begin
				if (num==0)
					begin 
						if (mem!=(4+(slvl*4))-1) slvl=slvl+1; 
						else 
                            begin 
                                slvl=0; 
                                lvl=lvl+1; 
                            end 
					end
				if (lvl==(filt+1))  
					begin
						lvl = 0;
						TOPlvl_conv = TOPlvl_conv+1'b1;
					end
			end
	end

assign memstartw_lvl = memstartw+lvl+slvl*(4*(filt+1))+num*(filt+1)*num_conv;
assign memstartzap_num = memstartzap+(((globmaxp_en==1)&&(lvl==filt))?(slvl*(4>>(num_conv>>1))+num):((conv_en==1)?(num*matrix2+slvl*matrix2*((4>>(num_conv>>1)))):((maxp_en==1)?num_maxp*(matrix2>>2):0)));
assign memstartp_lvl = memstartp+(lvl>>(num_conv>>1))*matrix2+((maxp_en==1)?num_maxp*matrix2:0);   //new!
	
assign re_p = (conv_en==1)?re_conv:((maxp_en==1)?re_maxp:((dense_en==1)?re_p_dense:((result_en==1)?re_p_res:0)));
assign re_w = (conv_en==1)?re_wb_conv:((dense_en==1)?re_w_dense:0);
assign read_addressp = (conv_en==1)?read_addressp_conv:((maxp_en==1)?read_addressp_maxp:((dense_en==1)?read_addressp_dense:((result_en==1)?read_addressp_res:0)));
assign we_p = (step==1)?we_p_zagr:((conv_en==1)?we_conv:((maxp_en==1)?we_maxp:((dense_en==1)?we_dense:0)));
assign dp = (step==1)?dp_zagr:((conv_en==1)?dp_conv:((maxp_en==1)?dp_maxp:((dense_en==1)?dp_dense:0)));
assign write_addressp = (step==1)?write_addressp_zagr:((conv_en==1)?write_addressp_conv:((maxp_en==1)?write_addressp_maxp:((dense_en==1)?write_addressp_dense:0)));
assign read_addressw = (conv_en==1)?read_addressw_conv:((dense_en==1)?read_addressw_dense:0);

assign matrix2 = matrix*matrix;

assign p11 = (conv_en==1)?p1_c:((dense_en==1)?p11_d:0);  //center
assign p12 = (conv_en==1)?p2_c:((dense_en==1)?p12_d:0);  //right
assign p13 = (conv_en==1)?p3_c:((dense_en==1)?p13_d:0);  //left
assign p14 = (conv_en==1)?p4_c:((dense_en==1)?p14_d:0);  //downleft
assign p15 = (conv_en==1)?p5_c:((dense_en==1)?p15_d:0);  //upright
assign p16 = (conv_en==1)?p6_c:((dense_en==1)?p16_d:0);  //down
assign p17 = (conv_en==1)?p7_c:((dense_en==1)?p17_d:0);  //up
assign p18 = (conv_en==1)?p8_c:((dense_en==1)?p18_d:0);  //downright
assign p19 = (conv_en==1)?p9_c:((dense_en==1)?p19_d:0);  //upleft

assign w11 = (conv_en==1)?w11_c:((dense_en==1)?w11_d:0);
assign w12 = (conv_en==1)?w12_c:((dense_en==1)?w12_d:0);
assign w13 = (conv_en==1)?w13_c:((dense_en==1)?w13_d:0);
assign w14 = (conv_en==1)?w14_c:((dense_en==1)?w14_d:0);
assign w15 = (conv_en==1)?w15_c:((dense_en==1)?w15_d:0);
assign w16 = (conv_en==1)?w16_c:((dense_en==1)?w16_d:0);
assign w17 = (conv_en==1)?w17_c:((dense_en==1)?w17_d:0);
assign w18 = (conv_en==1)?w18_c:((dense_en==1)?w18_d:0);
assign w19 = (conv_en==1)?w19_c:((dense_en==1)?w19_d:0);

assign TOPlvl = TOPlvl_conv+TOPlvl_maxp;


assign go_conv = (conv_en==1)?go_conv_TOP:((dense_en==1)?go_dense:0);

assign RESULT = (STOP)?res_out:4'b1111;

initial num = 0;
always @(posedge STOP_conv) 
    begin 
        if (num!=(4 >> (num_conv >> 1))-1) num = num+1; 
        else num = 0; 
    end

endmodule
