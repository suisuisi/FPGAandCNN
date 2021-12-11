module pre_v2(clk, rst_n, start, data, end_pre, output_data, x, y, i, j ,data_req);
  input clk;
  input rst_n;
  input [15:0] data;
  input start;
  output reg end_pre;
  output reg [10:0] output_data;
  input [9:0] x, y;
  output reg [4:0] i, j;
  output data_req;
  

reg [16:0] sr_data_0_0;
reg wr_sr_data_0;
wire [10:0] output_data_0_0;
reg [16:0] sr_data_1_0;
reg wr_sr_data_1;
wire [10:0] output_data_1_0;
reg [16:0] sr_data_2_0;
reg wr_sr_data_2;
wire [10:0] output_data_2_0;
reg [16:0] sr_data_3_0;
reg wr_sr_data_3;
wire [10:0] output_data_3_0;
reg [16:0] sr_data_4_0;
reg wr_sr_data_4;
wire [10:0] output_data_4_0;
reg [16:0] sr_data_5_0;
reg wr_sr_data_5;
wire [10:0] output_data_5_0;
reg [16:0] sr_data_6_0;
reg wr_sr_data_6;
wire [10:0] output_data_6_0;
reg [16:0] sr_data_7_0;
reg wr_sr_data_7;
wire [10:0] output_data_7_0;
reg [16:0] sr_data_8_0;
reg wr_sr_data_8;
wire [10:0] output_data_8_0;
reg [16:0] sr_data_9_0;
reg wr_sr_data_9;
wire [10:0] output_data_9_0;
reg [16:0] sr_data_10_0;
reg wr_sr_data_10;
wire [10:0] output_data_10_0;
reg [16:0] sr_data_11_0;
reg wr_sr_data_11;
wire [10:0] output_data_11_0;
reg [16:0] sr_data_12_0;
reg wr_sr_data_12;
wire [10:0] output_data_12_0;
reg [16:0] sr_data_13_0;
reg wr_sr_data_13;
wire [10:0] output_data_13_0;
reg [16:0] sr_data_14_0;
reg wr_sr_data_14;
wire [10:0] output_data_14_0;
reg [16:0] sr_data_15_0;
reg wr_sr_data_15;
wire [10:0] output_data_15_0;
reg [16:0] sr_data_16_0;
reg wr_sr_data_16;
wire [10:0] output_data_16_0;
reg [16:0] sr_data_17_0;
reg wr_sr_data_17;
wire [10:0] output_data_17_0;
reg [16:0] sr_data_18_0;
reg wr_sr_data_18;
wire [10:0] output_data_18_0;
reg [16:0] sr_data_19_0;
reg wr_sr_data_19;
wire [10:0] output_data_19_0;
reg [16:0] sr_data_20_0;
reg wr_sr_data_20;
wire [10:0] output_data_20_0;
reg [16:0] sr_data_21_0;
reg wr_sr_data_21;
wire [10:0] output_data_21_0;
reg [16:0] sr_data_22_0;
reg wr_sr_data_22;
wire [10:0] output_data_22_0;
reg [16:0] sr_data_23_0;
reg wr_sr_data_23;
wire [10:0] output_data_23_0;
reg [16:0] sr_data_24_0;
reg wr_sr_data_24;
wire [10:0] output_data_24_0;
reg [16:0] sr_data_25_0;
reg wr_sr_data_25;
wire [10:0] output_data_25_0;
reg [16:0] sr_data_26_0;
reg wr_sr_data_26;
wire [10:0] output_data_26_0;
reg [16:0] sr_data_27_0;
reg wr_sr_data_27;
wire [10:0] output_data_27_0;

reg [10:0] R;
reg [10:0] G;
reg [10:0] B;
reg [17:0] gray;
  
always @(posedge clk or negedge rst_n)	
	begin
	if ( !rst_n )
		begin
			i = 5'b0;
			j = 5'b0;
			sr_data_0_0 = 30'd0;
			sr_data_1_0 = 30'd0;
			sr_data_2_0 = 30'd0;
			sr_data_3_0 = 30'd0;
			sr_data_4_0 = 30'd0;
			sr_data_5_0 = 30'd0;
			sr_data_6_0 = 30'd0;
			sr_data_7_0 = 30'd0;
			sr_data_8_0 = 30'd0;
			sr_data_9_0 = 30'd0;
			sr_data_10_0 = 30'd0;
			sr_data_11_0 = 30'd0;
			sr_data_12_0 = 30'd0;
			sr_data_13_0 = 30'd0;
			sr_data_14_0 = 30'd0;
			sr_data_15_0 = 30'd0;
			sr_data_16_0 = 30'd0;
			sr_data_17_0 = 30'd0;
			sr_data_18_0 = 30'd0;
			sr_data_19_0 = 30'd0;
			sr_data_20_0 = 30'd0;
			sr_data_21_0 = 30'd0;
			sr_data_22_0 = 30'd0;
			sr_data_23_0 = 30'd0;
			sr_data_24_0 = 30'd0;
			sr_data_25_0 = 30'd0;
			sr_data_26_0 = 30'd0;
			sr_data_27_0 = 30'd0;
			end_pre = 1'b0;
			gray = 18'd0;
		end
	else
    begin
      if (start)
        begin
			R = 11'd0;
			R[9] = data[15];
			R[8] = data[14];
			R[7] = data[13];
			R[6] = data[12];
			R[5] = data[11];

			G = 11'd0;
			G[9] = data[10];
			G[8] = data[9];
			G[7] = data[8];
			G[6] = data[7];
			G[5] = data[6];
			G[4] = data[5];

			B = 11'd0;
			B[9] = data[4];
			B[8] = data[3];
			B[7] = data[2];
			B[6] = data[1];
			B[5] = data[0];
				  

         gray = 3*B + 8*G + 5*R;
         gray = gray >> 4;
		 
			if ((x>(10'd47+(0*8)))&&(x<(10'd56+(0*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_0_0 = sr_data_0_0 + gray;
				end
			if ((x>(10'd47+(1*8)))&&(x<(10'd56+(1*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_1_0 = sr_data_1_0 + gray;
				end
			if ((x>(10'd47+(2*8)))&&(x<(10'd56+(2*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_2_0 = sr_data_2_0 + gray;
				end
			if ((x>(10'd47+(3*8)))&&(x<(10'd56+(3*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_3_0 = sr_data_3_0 + gray;
				end
			if ((x>(10'd47+(4*8)))&&(x<(10'd56+(4*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_4_0 = sr_data_4_0 + gray;
				end
			if ((x>(10'd47+(5*8)))&&(x<(10'd56+(5*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_5_0 = sr_data_5_0 + gray;
				end
			if ((x>(10'd47+(6*8)))&&(x<(10'd56+(6*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_6_0 = sr_data_6_0 + gray;
				end
			if ((x>(10'd47+(7*8)))&&(x<(10'd56+(7*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_7_0 = sr_data_7_0 + gray;
				end
			if ((x>(10'd47+(8*8)))&&(x<(10'd56+(8*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_8_0 = sr_data_8_0 + gray;
				end
			if ((x>(10'd47+(9*8)))&&(x<(10'd56+(9*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_9_0 = sr_data_9_0 + gray;
				end
			if ((x>(10'd47+(10*8)))&&(x<(10'd56+(10*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_10_0 = sr_data_10_0 + gray;
				end
			if ((x>(10'd47+(11*8)))&&(x<(10'd56+(11*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_11_0 = sr_data_11_0 + gray;
				end
			if ((x>(10'd47+(12*8)))&&(x<(10'd56+(12*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_12_0 = sr_data_12_0 + gray;
				end
			if ((x>(10'd47+(13*8)))&&(x<(10'd56+(13*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_13_0 = sr_data_13_0 + gray;
				end
			if ((x>(10'd47+(14*8)))&&(x<(10'd56+(14*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_14_0 = sr_data_14_0 + gray;
				end
			if ((x>(10'd47+(15*8)))&&(x<(10'd56+(15*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_15_0 = sr_data_15_0 + gray;
				end
			if ((x>(10'd47+(16*8)))&&(x<(10'd56+(16*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_16_0 = sr_data_16_0 + gray;
				end
			if ((x>(10'd47+(17*8)))&&(x<(10'd56+(17*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_17_0 = sr_data_17_0 + gray;
				end
			if ((x>(10'd47+(18*8)))&&(x<(10'd56+(18*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_18_0 = sr_data_18_0 + gray;
				end
			if ((x>(10'd47+(19*8)))&&(x<(10'd56+(19*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_19_0 = sr_data_19_0 + gray;
				end
			if ((x>(10'd47+(20*8)))&&(x<(10'd56+(20*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_20_0 = sr_data_20_0 + gray;
				end
			if ((x>(10'd47+(21*8)))&&(x<(10'd56+(21*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_21_0 = sr_data_21_0 + gray;
				end
			if ((x>(10'd47+(22*8)))&&(x<(10'd56+(22*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_22_0 = sr_data_22_0 + gray;
				end
			if ((x>(10'd47+(23*8)))&&(x<(10'd56+(23*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_23_0 = sr_data_23_0 + gray;
				end
			if ((x>(10'd47+(24*8)))&&(x<(10'd56+(24*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_24_0 = sr_data_24_0 + gray;
				end
			if ((x>(10'd47+(25*8)))&&(x<(10'd56+(25*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_25_0 = sr_data_25_0 + gray;
				end
			if ((x>(10'd47+(26*8)))&&(x<(10'd56+(26*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_26_0 = sr_data_26_0 + gray;
				end
			if ((x>(10'd47+(27*8)))&&(x<(10'd56+(27*8)))&&(y>(10'd7+(j*8)))&&(y<(10'd16+(j*8)))) 
				begin		
				  sr_data_27_0 = sr_data_27_0 + gray;
				end
			if ((x==(10'd57+(0*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_0_0; i=0; end
			if ((x>(10'd59+(0*8)))&&(x<(10'd65+(0*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_0=1'b1; 
			else wr_sr_data_0 = 1'b0;
			if ((x==(10'd57+(1*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_1_0; i=1; end
			if ((x>(10'd59+(1*8)))&&(x<(10'd65+(1*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_1=1'b1; 
			else wr_sr_data_1 = 1'b0;
			if ((x==(10'd57+(2*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_2_0; i=2; end
			if ((x>(10'd59+(2*8)))&&(x<(10'd65+(2*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_2=1'b1; 
			else wr_sr_data_2 = 1'b0;
			if ((x==(10'd57+(3*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_3_0; i=3; end
			if ((x>(10'd59+(3*8)))&&(x<(10'd65+(3*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_3=1'b1; 
			else wr_sr_data_3 = 1'b0;
			if ((x==(10'd57+(4*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_4_0; i=4; end
			if ((x>(10'd59+(4*8)))&&(x<(10'd65+(4*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_4=1'b1; 
			else wr_sr_data_4 = 1'b0;
			if ((x==(10'd57+(5*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_5_0; i=5; end
			if ((x>(10'd59+(5*8)))&&(x<(10'd65+(5*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_5=1'b1; 
			else wr_sr_data_5 = 1'b0;
			if ((x==(10'd57+(6*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_6_0; i=6; end
			if ((x>(10'd59+(6*8)))&&(x<(10'd65+(6*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_6=1'b1; 
			else wr_sr_data_6 = 1'b0;
			if ((x==(10'd57+(7*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_7_0; i=7; end
			if ((x>(10'd59+(7*8)))&&(x<(10'd65+(7*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_7=1'b1; 
			else wr_sr_data_7 = 1'b0;
			if ((x==(10'd57+(8*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_8_0; i=8; end
			if ((x>(10'd59+(8*8)))&&(x<(10'd65+(8*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_8=1'b1; 
			else wr_sr_data_8 = 1'b0;
			if ((x==(10'd57+(9*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_9_0; i=9; end
			if ((x>(10'd59+(9*8)))&&(x<(10'd65+(9*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_9=1'b1; 
			else wr_sr_data_9 = 1'b0;
			if ((x==(10'd57+(10*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_10_0; i=10; end
			if ((x>(10'd59+(10*8)))&&(x<(10'd65+(10*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_10=1'b1; 
			else wr_sr_data_10 = 1'b0;
			if ((x==(10'd57+(11*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_11_0; i=11; end
			if ((x>(10'd59+(11*8)))&&(x<(10'd65+(11*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_11=1'b1; 
			else wr_sr_data_11 = 1'b0;
			if ((x==(10'd57+(12*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_12_0; i=12; end
			if ((x>(10'd59+(12*8)))&&(x<(10'd65+(12*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_12=1'b1; 
			else wr_sr_data_12 = 1'b0;
			if ((x==(10'd57+(13*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_13_0; i=13; end
			if ((x>(10'd59+(13*8)))&&(x<(10'd65+(13*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_13=1'b1; 
			else wr_sr_data_13 = 1'b0;
			if ((x==(10'd57+(14*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_14_0; i=14; end
			if ((x>(10'd59+(14*8)))&&(x<(10'd65+(14*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_14=1'b1; 
			else wr_sr_data_14 = 1'b0;
			if ((x==(10'd57+(15*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_15_0; i=15; end
			if ((x>(10'd59+(15*8)))&&(x<(10'd65+(15*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_15=1'b1; 
			else wr_sr_data_15 = 1'b0;
			if ((x==(10'd57+(16*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_16_0; i=16; end
			if ((x>(10'd59+(16*8)))&&(x<(10'd65+(16*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_16=1'b1; 
			else wr_sr_data_16 = 1'b0;
			if ((x==(10'd57+(17*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_17_0; i=17; end
			if ((x>(10'd59+(17*8)))&&(x<(10'd65+(17*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_17=1'b1; 
			else wr_sr_data_17 = 1'b0;
			if ((x==(10'd57+(18*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_18_0; i=18; end
			if ((x>(10'd59+(18*8)))&&(x<(10'd65+(18*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_18=1'b1; 
			else wr_sr_data_18 = 1'b0;
			if ((x==(10'd57+(19*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_19_0; i=19; end
			if ((x>(10'd59+(19*8)))&&(x<(10'd65+(19*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_19=1'b1; 
			else wr_sr_data_19 = 1'b0;
			if ((x==(10'd57+(20*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_20_0; i=20; end
			if ((x>(10'd59+(20*8)))&&(x<(10'd65+(20*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_20=1'b1; 
			else wr_sr_data_20 = 1'b0;
			if ((x==(10'd57+(21*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_21_0; i=21; end
			if ((x>(10'd59+(21*8)))&&(x<(10'd65+(21*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_21=1'b1; 
			else wr_sr_data_21 = 1'b0;
			if ((x==(10'd57+(22*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_22_0; i=22; end
			if ((x>(10'd59+(22*8)))&&(x<(10'd65+(22*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_22=1'b1; 
			else wr_sr_data_22 = 1'b0;
			if ((x==(10'd57+(23*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_23_0; i=23; end
			if ((x>(10'd59+(23*8)))&&(x<(10'd65+(23*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_23=1'b1; 
			else wr_sr_data_23 = 1'b0;
			if ((x==(10'd57+(24*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_24_0; i=24; end
			if ((x>(10'd59+(24*8)))&&(x<(10'd65+(24*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_24=1'b1; 
			else wr_sr_data_24 = 1'b0;
			if ((x==(10'd57+(25*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_25_0; i=25; end
			if ((x>(10'd59+(25*8)))&&(x<(10'd65+(25*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_25=1'b1; 
			else wr_sr_data_25 = 1'b0;
			if ((x==(10'd57+(26*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_26_0; i=26; end
			if ((x>(10'd59+(26*8)))&&(x<(10'd65+(26*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_26=1'b1; 
			else wr_sr_data_26 = 1'b0;
			if ((x==(10'd57+(27*8)))&&(y==(10'd15+(j*8))))	begin output_data=output_data_27_0; i=27; end
			if ((x>(10'd59+(27*8)))&&(x<(10'd65+(27*8)))&&(y==(10'd15+(j*8))))	wr_sr_data_27=1'b1; 
			else wr_sr_data_27 = 1'b0;
			
			if ((x==10'd319) && (y==(10'd15+(j*8))))
				begin
					if (j >= 5'd27) 
					begin	
					  //j = 5'd0;
					  end_pre = 1'b1;
					end
					else j = j + 1'b1;
					sr_data_0_0 = 30'd0;
					sr_data_1_0 = 30'd0;
					sr_data_2_0 = 30'd0;
					sr_data_3_0 = 30'd0;
					sr_data_4_0 = 30'd0;
					sr_data_5_0 = 30'd0;
					sr_data_6_0 = 30'd0;
					sr_data_7_0 = 30'd0;
					sr_data_8_0 = 30'd0;
					sr_data_9_0 = 30'd0;
					sr_data_10_0 = 30'd0;
					sr_data_11_0 = 30'd0;
					sr_data_12_0 = 30'd0;
					sr_data_13_0 = 30'd0;
					sr_data_14_0 = 30'd0;
					sr_data_15_0 = 30'd0;
					sr_data_16_0 = 30'd0;
					sr_data_17_0 = 30'd0;
					sr_data_18_0 = 30'd0;
					sr_data_19_0 = 30'd0;
					sr_data_20_0 = 30'd0;
					sr_data_21_0 = 30'd0;
					sr_data_22_0 = 30'd0;
					sr_data_23_0 = 30'd0;
					sr_data_24_0 = 30'd0;
					sr_data_25_0 = 30'd0;
					sr_data_26_0 = 30'd0;
					sr_data_27_0 = 30'd0;
				end
        end
		else
			   begin
						i=5'b0;
						j=5'b0;
						sr_data_0_0 = 30'd0;
						sr_data_1_0 = 30'd0;
						sr_data_2_0 = 30'd0;
						sr_data_3_0 = 30'd0;
						sr_data_4_0 = 30'd0;
						sr_data_5_0 = 30'd0;
						sr_data_6_0 = 30'd0;
						sr_data_7_0 = 30'd0;
						sr_data_8_0 = 30'd0;
						sr_data_9_0 = 30'd0;
						sr_data_10_0 = 30'd0;
						sr_data_11_0 = 30'd0;
						sr_data_12_0 = 30'd0;
						sr_data_13_0 = 30'd0;
						sr_data_14_0 = 30'd0;
						sr_data_15_0 = 30'd0;
						sr_data_16_0 = 30'd0;
						sr_data_17_0 = 30'd0;
						sr_data_18_0 = 30'd0;
						sr_data_19_0 = 30'd0;
						sr_data_20_0 = 30'd0;
						sr_data_21_0 = 30'd0;
						sr_data_22_0 = 30'd0;
						sr_data_23_0 = 30'd0;
						sr_data_24_0 = 30'd0;
						sr_data_25_0 = 30'd0;
						sr_data_26_0 = 30'd0;
						sr_data_27_0 = 30'd0;
						end_pre = 1'b0;
						gray = 18'd0;
				end
			
    end
end

        
assign data_req = wr_sr_data_0 | wr_sr_data_1 | wr_sr_data_2 | wr_sr_data_3 | wr_sr_data_4 | wr_sr_data_5 | wr_sr_data_6 | wr_sr_data_7 | wr_sr_data_8 | wr_sr_data_9 | wr_sr_data_10 | wr_sr_data_11 | wr_sr_data_12 | wr_sr_data_13 | wr_sr_data_14 | wr_sr_data_15 | wr_sr_data_16 | wr_sr_data_17 | wr_sr_data_18 | wr_sr_data_19 | wr_sr_data_20 | wr_sr_data_21 | wr_sr_data_22 | wr_sr_data_23 | wr_sr_data_24 | wr_sr_data_25 | wr_sr_data_26 | wr_sr_data_27;
assign output_data_0_0 = sr_data_0_0 >> 6;
assign output_data_1_0 = sr_data_1_0 >> 6;
assign output_data_2_0 = sr_data_2_0 >> 6;
assign output_data_3_0 = sr_data_3_0 >> 6;
assign output_data_4_0 = sr_data_4_0 >> 6;
assign output_data_5_0 = sr_data_5_0 >> 6;
assign output_data_6_0 = sr_data_6_0 >> 6;
assign output_data_7_0 = sr_data_7_0 >> 6;
assign output_data_8_0 = sr_data_8_0 >> 6;
assign output_data_9_0 = sr_data_9_0 >> 6;
assign output_data_10_0 = sr_data_10_0 >> 6;
assign output_data_11_0 = sr_data_11_0 >> 6;
assign output_data_12_0 = sr_data_12_0 >> 6;
assign output_data_13_0 = sr_data_13_0 >> 6;
assign output_data_14_0 = sr_data_14_0 >> 6;
assign output_data_15_0 = sr_data_15_0 >> 6;
assign output_data_16_0 = sr_data_16_0 >> 6;
assign output_data_17_0 = sr_data_17_0 >> 6;
assign output_data_18_0 = sr_data_18_0 >> 6;
assign output_data_19_0 = sr_data_19_0 >> 6;
assign output_data_20_0 = sr_data_20_0 >> 6;
assign output_data_21_0 = sr_data_21_0 >> 6;
assign output_data_22_0 = sr_data_22_0 >> 6;
assign output_data_23_0 = sr_data_23_0 >> 6;
assign output_data_24_0 = sr_data_24_0 >> 6;
assign output_data_25_0 = sr_data_25_0 >> 6;
assign output_data_26_0 = sr_data_26_0 >> 6;
assign output_data_27_0 = sr_data_27_0 >> 6;

endmodule
