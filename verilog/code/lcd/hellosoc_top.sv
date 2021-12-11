module hellosoc_top(
	input tft_sdo, 
	output wire tft_sck, 
	output wire tft_sdi, 
	output wire tft_dc, 
	output wire tft_reset, 
	output wire tft_cs,
	input			rst_n,
	input 		clk_sdram,
	input			wr_fifo,
	input [15:0] sdram_data,
	input tft_clk,
	output [11:0] output_rdusedw,
	output fbClk,
	output [4:0] r,
	output [5:0] g,
	output [4:0] b,
	output reg start_28,
	input [3:0] RESULT
	);
	reg start;
	wire [15:0] q;
	assign r={q[15],q[14],q[13],q[12],q[11]};
	assign g={q[10],q[9],q[8],q[7],q[6],q[5]};
	assign b={q[4],q[3],q[2],q[1],q[0]};
	
	// X,Y calc
	reg [9:0] x;
	reg [9:0] y;
	
	

	fifo_big fifo_tft
(
	.aclr				( !rst_n			),
	.data				( sdram_data	),
	.rdclk              ( fbClk		),
	.rdreq              ( start 	),
	.wrclk              ( clk_sdram			),
	.wrreq              ( wr_fifo  ),
	.q                  ( q					),
	.rdempty            ( 					),
	.rdusedw            ( output_rdusedw	),
	.wrfull             ( 					),
	.wrusedw            ( 					)
);
	

	wire valid;
	reg [9:0] str;
	reg [8:0] x_out;
	reg [8:0] y_out;
	initial x_out=9'd239;
	initial y_out=9'd319;
	always @ (posedge clk_sdram or negedge rst_n) 
	begin
		if ( !rst_n )
			begin
					x=10'd319;
					y=10'd239;
					start=1'b0;
					start_28=1'b0;
			end
		else
			begin
				if (wr_fifo)
					begin
						if (x==10'd319) 
						begin
							x<=10'd0;
							if (y==10'd239) 
								begin
									y<=10'd0;
								end
							else y<=y+1'b1;
						end
						else x<=x+1'b1;
					end
					
					
				if ((x_out==9'd13)&&(y_out==9'd47)) start=1'b1;
				if ((x_out==9'd260)&&(y_out==9'd47)) start_28=1'b1;
			end	
	end	
	

	always @(posedge fbClk or negedge rst_n) begin
		if ( !rst_n )
			begin
			end
		else
			begin
				if (x_out==9'd319) 
						begin
							x_out<=9'd0;
							if (y_out==9'd239) 
								begin
									y_out<=9'd0;
								end
							else y_out<=y_out+1'b1;
						end
				else x_out<=x_out+1'b1;		
			end
		end
		
		
	
	//rbg
	//16'b0 00 0000 0 0000 0 100 - green
	//16'b0 00 0000 0 1000 0 000 - blue
	//16'b0 00 1000 0 0000 0 000 - red
	wire [15:0] currentPixel;
	wire RES_0,RES_1,RES_2,RES_3,RES_4,RES_5,RES_6,RES_7,RES_8,RES_9,RES_10;
	assign RES_0=(((x_out>250)&&(x_out<254)&&(y_out>5)&&(y_out<43))||((x_out>274)&&(x_out<278)&&(y_out>5)&&(y_out<43))||((x_out>253)&&(x_out<275)&&(y_out>5)&&(y_out<9))||((x_out>253)&&(x_out<275)&&(y_out>39)&&(y_out<43)))?1'b1:1'b0;
	
	assign RES_1=(((x_out>274)&&(x_out<278)&&(y_out>5)&&(y_out<43))||((y_out>5)&&(y_out<26)&&((x_out==240-y_out+43)||(x_out==240-y_out-1+43)||(x_out==240-y_out-2+43)||(x_out==240-y_out-3+43))))?1'b1:1'b0;
	
	assign RES_2=(((x_out>250)&&(x_out<254)&&(y_out>22)&&(y_out<43))||((x_out>274)&&(x_out<278)&&(y_out>5)&&(y_out<26))||((x_out>250)&&(x_out<275)&&(y_out>5)&&(y_out<9))||((x_out>253)&&(x_out<278)&&(y_out>39)&&(y_out<43))||((x_out>253)&&(x_out<275)&&(y_out>22)&&(y_out<26)))?1'b1:1'b0;
	assign RES_3=(((x_out>274)&&(x_out<278)&&(y_out>5)&&(y_out<43))||((x_out>250)&&(x_out<275)&&(y_out>5)&&(y_out<9))||((x_out>250)&&(x_out<275)&&(y_out>39)&&(y_out<43))||((x_out>250)&&(x_out<275)&&(y_out>22)&&(y_out<26)))?1'b1:1'b0;
	assign RES_4=(((x_out>250)&&(x_out<254)&&(y_out>5)&&(y_out<26))||((x_out>274)&&(x_out<278)&&(y_out>5)&&(y_out<43))||((x_out>253)&&(x_out<275)&&(y_out>22)&&(y_out<26)))?1'b1:1'b0;
	assign RES_5=(((x_out>250)&&(x_out<254)&&(y_out>5)&&(y_out<26))||((x_out>274)&&(x_out<278)&&(y_out>22)&&(y_out<43))||((x_out>253)&&(x_out<278)&&(y_out>5)&&(y_out<9))||((x_out>250)&&(x_out<275)&&(y_out>39)&&(y_out<43))||((x_out>253)&&(x_out<275)&&(y_out>22)&&(y_out<26)))?1'b1:1'b0;
	assign RES_6=(((x_out>250)&&(x_out<254)&&(y_out>5)&&(y_out<43))||((x_out>274)&&(x_out<278)&&(y_out>22)&&(y_out<43))||((x_out>253)&&(x_out<278)&&(y_out>5)&&(y_out<9))||((x_out>253)&&(x_out<275)&&(y_out>39)&&(y_out<43))||((x_out>253)&&(x_out<275)&&(y_out>22)&&(y_out<26)))?1'b1:1'b0;
	assign RES_7=(((x_out>274)&&(x_out<278)&&(y_out>5)&&(y_out<43))||((x_out>250)&&(x_out<275)&&(y_out>5)&&(y_out<9)))?1'b1:1'b0;
	assign RES_8=(((x_out>250)&&(x_out<254)&&(y_out>5)&&(y_out<43))||((x_out>274)&&(x_out<278)&&(y_out>5)&&(y_out<43))||((x_out>253)&&(x_out<275)&&(y_out>5)&&(y_out<9))||((x_out>253)&&(x_out<275)&&(y_out>39)&&(y_out<43))||((x_out>253)&&(x_out<275)&&(y_out>22)&&(y_out<26)))?1'b1:1'b0;
	assign RES_9=(((x_out>250)&&(x_out<254)&&(y_out>5)&&(y_out<26))||((x_out>274)&&(x_out<278)&&(y_out>5)&&(y_out<43))||((x_out>253)&&(x_out<275)&&(y_out>5)&&(y_out<9))||((x_out>250)&&(x_out<275)&&(y_out>39)&&(y_out<43))||((x_out>253)&&(x_out<275)&&(y_out>22)&&(y_out<26)))?1'b1:1'b0;
	assign RES_10=((x_out>254)&&(x_out<274)&&(y_out>22)&&(y_out<26))?1'b1:1'b0;
	assign currentPixel = ((x_out<288)&&(x_out>240)&&(y_out>=0)&&(y_out<47))?( ((RESULT==4'd10)?((RES_10)?16'h0:16'hFFFF):((RESULT==4'd0)?((RES_0)?16'h0:16'hFFFF):((RESULT==4'd1)?((RES_1)?16'h0:16'hFFFF):((RESULT==4'd2)?((RES_2)?16'h0:16'hFFFF):((RESULT==4'd3)?((RES_3)?16'h0:16'hFFFF):((RESULT==4'd4)?((RES_4)?16'h0:16'hFFFF):((RESULT==4'd5)?((RES_5)?16'h0:16'hFFFF):((RESULT==4'd6)?((RES_6)?16'h0:16'hFFFF):((RESULT==4'd7)?((RES_7)?16'h0:16'hFFFF):((RESULT==4'd8)?((RES_8)?16'h0:16'hFFFF):((RESULT==4'd9)?((RES_9)?16'h0:16'hFFFF):(16'hFFFF))))))))))))                            ):(((((y_out==9'd56)||(y_out==9'd40))&&(((x_out>=9'd288)&&(x_out<=9'd320))||((x_out>=9'd0)&&(x_out<=9'd192))))||(x_out==9'd192)||(x_out==288))?16'b0001111100000000:{g[2],g[1],g[0],r[4],r[3],r[2],r[1],r[0],b[4],b[3],b[2],b[1],b[0],g[5],g[4],g[3]})/*:((RESULT==4'd10)?((RES_10)?16'h0:16'hFFFF):((RESULT==4'd0)?((RES_0)?16'h0:16'hFFFF):((RESULT==4'd1)?((RES_1)?16'h0:16'hFFFF):((RESULT==4'd2)?((RES_2)?16'h0:16'hFFFF):((RESULT==4'd3)?((RES_3)?16'h0:16'hFFFF):((RESULT==4'd4)?((RES_4)?16'h0:16'hFFFF):((RESULT==4'd5)?((RES_5)?16'h0:16'hFFFF):((RESULT==4'd6)?((RES_6)?16'h0:16'hFFFF):((RESULT==4'd7)?((RES_7)?16'h0:16'hFFFF):((RESULT==4'd8)?((RES_8)?16'h0:16'hFFFF):((RESULT==4'd9)?((RES_9)?16'h0:16'hFFFF):(16'hFFFF))))))))))))*/; 

	// *************************** TFT Module
	tft_ili9341 #(.INPUT_CLK_MHZ(100)) tft(tft_clk, tft_sdo, tft_sck, tft_sdi, tft_dc, tft_reset, tft_cs, currentPixel, fbClk);

endmodule
