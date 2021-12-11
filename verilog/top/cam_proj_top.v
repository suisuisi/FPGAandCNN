// Based on verilog by shvlad for camera connection (email: shvladspb@gmail.com)
// https://habrahabr.ru/post/283488/
// synopsys translate_off
`timescale 1 ns / 1 ns
// synopsys translate_on

module cam_proj_top
(
	//clocks & reset
	input wire clk50,             
	input wire rst,
	input wire start_gray_kn,
	//OV7670
	input wire [7:0] data_cam,
	input wire VSYNC_cam,
	input wire HREF_cam,
	input wire PCLK_cam,
	output wire XCLK_cam,
	output reg res_cam,
	output reg on_off_cam,
	output wire sioc,
	output wire siod,
	//VGA
	output wire [4:0] r,  
	output wire [5:0] g,
	output wire [4:0] b, 	
	//SDRAM
	output wire cs_n,
	output wire ras_n,
	output wire cas_n,
	output wire we_n,
	output wire [1:0] dqm,
	output wire [11:0] sd_addr,
	output wire [1:0] ba,
	output wire Cke,
	inout wire [15:0] sd_data, 
	output wire sdram_clk, 
	//TFT
	input tft_sdo,
	output wire tft_sck, 
	output wire tft_sdi, 
	output wire tft_dc, 
	output wire tft_reset, 
	output wire tft_cs,
	//LED
	output [7:0] LED
);

wire fbClk;
wire rst_n;
wire clk25;
wire clk100;
wire clk24;
wire clk143; 
wire [15:0] input_fifo_to_sdram;

assign rst_n = rst;
assign sdram_clk = clk143;

wire locked;
wire [11:0] output_rdusedw_TFT;
wire start_image;
wire [3:0] RESULT;
reg GO_NEIROSET;
wire end_neiroset;
reg [3:0] RESULT_2;

wire ctrl_busy;
wire [23:0] wr_addr;
wire wr_enable;
wire [15:0] rd_data;
reg [23:0] rd_addr;
wire rd_ready;
reg ready;


//LED
assign LED[0] = wr_enable;
assign LED[1] = !wr_enable;
assign LED[2] = 1'b0;
assign LED[3] = 1'b0;
assign LED[4] = 1'b0;
assign LED[5] = 1'b0;
assign LED[6] = 1'b0;
assign LED[7] = 1'b0;


// Clocks

pll pll_for_sdram_0
(
	.areset   ( !rst_n ),
	.inclk0   ( clk50 ),
	.c0       ( clk100 ),
	.c2       ( clk25 ),
	.c3       ( clk24 ),
	.locked   ( locked )
);

pll_for_disp pll2
(
	.areset   ( !rst_n ),
	.inclk0   ( clk50 ),
	.c0       ( clk143 )

);

// Process data from camera

cam_wrp cam_wrp_0
(
	.rst_n                ( rst_n ),
	.data_cam             ( data_cam ),
	.HREF_cam             ( HREF_cam ),
	.PCLK_cam             ( PCLK_cam ),
	.ctrl_busy            ( ctrl_busy ),
	.input_fifo_to_sdram  ( input_fifo_to_sdram ),
	.addr_sdram           ( wr_addr ),
	.wr_enable            ( wr_enable )
);

assign XCLK_cam = clk24;

//TFT display

hellosoc_top TFT(
	.tft_sdo            ( tft_sdo ), 
	.tft_sck            ( tft_sck ), 
	.tft_sdi            ( tft_sdi ), 
	.tft_dc             ( tft_dc ), 
	.tft_reset          ( tft_reset ), 
	.tft_cs             ( tft_cs ),
	.rst_n              ( rst_n ),
	.clk_sdram          ( !rd_ready ),
	.wr_fifo            ( (!wr_enable) && ready ),
	.sdram_data         ( rd_data ),
	.tft_clk            ( clk100 ),
	.output_rdusedw     ( output_rdusedw_TFT ),
	.fbClk              ( fbClk ),
	.r                  ( r ),
	.g                  ( g ),
	.b                  ( b ),
	.start_28           ( start_image ),
	.RESULT             ( RESULT_2 )
);

reg [9:0] x_sdram;

always @(posedge rd_ready or negedge rst_n)
begin
	if (!rst_n)
	begin
		rd_addr = 320*201+3;
		ready=1'b1;
		x_sdram=0;
	end
	else
	begin
		if ((!wr_enable)&&(ready))
				begin
					if (rd_addr < 24'd76799) rd_addr = rd_addr + 1'b1;
					else rd_addr=24'd0;
					if (x_sdram<320) x_sdram=x_sdram+1'b1;
					else x_sdram=1;
				end
		if (x_sdram==320)
			begin
				if ((!wr_enable)&&(output_rdusedw_TFT<=3000)) begin ready=1'b1;  end
				else begin  ready=1'b0;  end
			end
	end
end

	
sdram_controller SDRAM(
	.wr_addr       (wr_addr),
	.wr_data       (input_fifo_to_sdram),
	.wr_enable     (wr_enable),
	.rd_addr       (rd_addr),
	.rd_data       (rd_data),
	.rd_ready      (rd_ready),
	.rd_enable     (!wr_enable),
	.busy          (ctrl_busy), 
	.rst_n         (rst_n), 
	.clk           (clk143),
	/* SDRAM SIDE */
	.addr          (sd_addr), 
	.bank_addr     (ba), 
	.data          (sd_data), 
	.clock_enable  (Cke),	 
	.cs_n          (cs_n), 
	.ras_n         (ras_n), 
	.cas_n         (cas_n), 
	.we_n          (we_n),	
	.data_mask_low (dqm[0]), 
	.data_mask_high(dqm[1]) 
);


reg start_gray;
wire end_gray;
reg [9:0] x_gray, y_gray;
wire [4:0] i_gray, j_gray;
wire [12:0] out_data_gray;
wire wrreq_gray;

pre_v2 grayscale(
	.clk           (fbClk), 
	.rst_n         (rst_n),
	.start         (start_gray), 
	.data          ({r,g,b}), 
	.end_pre       (end_gray), 
	.output_data   (out_data_gray), 
	.x             (x_gray), 
	.y             (y_gray), 
	.i             (i_gray), 
	.j             (j_gray) ,
	.data_req      (wrreq_gray)
);


TOP neiroset (
	.clk                (clk50),
	.GO                 (GO_NEIROSET),
	.RESULT             (RESULT),
	.we_database        (wrreq_gray),
	.dp_database        (out_data_gray),
	.address_p_database (j_gray*28+i_gray),
	.STOP               (end_neiroset)
);


always @(posedge fbClk or negedge rst_n) 
begin
	if ( !rst_n )
	begin
		x_gray = 10'd0;
		y_gray = 10'd0;
		start_gray = 1'b0;
	end
	else
	begin
		if (start_image)
			begin
				if (x_gray == 10'd319) 
				begin
					x_gray <= 10'd0;
					if (y_gray == 10'd239) 
					begin
						y_gray <= 10'd0;
					end
					else y_gray <= y_gray+1'b1;
				end
				else x_gray <= x_gray+1'b1;
			end
			
		if ((GO_NEIROSET == 1'b1) && (x_gray == 10'd47) && (y_gray == 10'd239)) start_gray = 1'b1;
		if (end_gray) start_gray = 1'b0;
	end
end	


always @(posedge clk50 or negedge rst_n) 
begin
	if ( !rst_n )
	begin
		RESULT_2 = 4'b1111;
		GO_NEIROSET = 1'b1;
	end
	else
	begin
		if (end_gray) begin GO_NEIROSET = 1'b0; end
		if (end_neiroset) begin GO_NEIROSET = 1'b1; RESULT_2 = RESULT; end
	end
end



// start camera inititalization
reg [2:0] strt;

always @(posedge clk25 or negedge rst_n)
	if (!rst_n)
		strt <= 3'h0;
	else
	begin
		if (locked)
			begin
				if ( &strt )
					strt	<= strt;
				else
					strt	<= strt + 1'h1;
			end
	end

// camera inititalization
camera_configure 
#(	
	.CLK_FREQ 	( 25000000 )
)
camera_configure_0
(
	.clk   ( clk25            ),	
	.start ( ( strt == 3'h6 ) ),
	.sioc  ( sioc             ),
	.siod  ( siod             ),
	.done  ( 			        )
);

// reset camera with overall reset from button
assign res_cam    = rst_n;
assign on_off_cam = !rst_n;

endmodule
