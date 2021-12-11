//author: shvlad
//email: shvladspb@gmail.com
`timescale 1 ns / 100 ps
//`default_nettype none
module fsm_global
(   
	input wire					clk,
	input wire 					rst_n,
	input wire					clk_cam,
//
	input wire		[9:0]		input_wrusedw,
	input wire		[11:0]		output_rdusedw,
	input wire					VSYNC_cam,
	input wire					HREF_cam,
	output wire					p_VSYNC_cam,
//от в входное фифо	
	input wire					wr_input_fifo,
	output wire					rd_input_fifo,
	
	output wire					wr_strobe,
	output wire					rd_strobe,
	output wire					write_strobe,
	output wire					read_strobe,
	output wire					wait_strobe,
	output wire					idle_strobe,
//sdram cntrl
	input wire					sd_ready,
	input wire					valid_data,
//
	input wire					STOP_capt
);

reg		[ 9: 0]			cnt_href;	
reg						sh_VSYNC_cam;
reg						sh1_VSYNC_cam;
reg						sh_HREF_cam;
reg						sh1_HREF_cam;
wire					p_HREF_cam;
reg						sh_clk_cam;
reg						sh1_clk_cam;
wire					p_clk_cam;
reg		[9 : 0]			cnt_rd_input_fifo;

//состояния автомата
enum reg [2:0]
	{	
		s1_idle				= 3'd1,
		s1_wait				= 3'd2,
		s1_rd				= 3'd3,
		s1_wr				= 3'd4
	}glob_cs,glob_ns;
	
// автомат считывания коэффициентов 
	always @(posedge clk or negedge rst_n)
			if ( !rst_n )		glob_cs	<= s1_idle;
			else				glob_cs	<= glob_ns;	

always_comb
	begin
		glob_ns = glob_cs;
		case ( glob_cs )
			s1_idle:				if ( rst_n )																glob_ns = s1_wait;
			s1_wait:				if ( ( input_wrusedw >= 10'd600 )  & sd_ready & (!STOP_capt) )				glob_ns = s1_wr;
									else if ( ( output_rdusedw <= 12'd1100 ) & p_clk_cam  & sd_ready )			glob_ns = s1_rd;
			s1_wr:					if ( !sd_ready )															glob_ns = s1_wait;
			s1_rd:					if ( !sd_ready )															glob_ns = s1_wait;			
			default																								glob_ns = s1_idle;
		endcase
	end

//выделяю фронты синхросигналов 	
	always @( posedge clk or negedge rst_n )
		if ( !rst_n )
			begin
				sh_VSYNC_cam	<= 1'h0;
				sh1_VSYNC_cam	<= 1'h0;				
				sh_HREF_cam 	<= 1'h0;
				sh1_HREF_cam	<= 1'h0;				
			end
		else
			begin
				sh_VSYNC_cam	<= VSYNC_cam;
				sh1_VSYNC_cam	<= sh_VSYNC_cam;
				sh_HREF_cam 	<= HREF_cam;	
				sh1_HREF_cam	<= sh_HREF_cam;				
			end
			
assign p_VSYNC_cam	= sh_VSYNC_cam & !sh1_VSYNC_cam;
assign p_HREF_cam   = sh_HREF_cam  & !sh1_HREF_cam ;

//выделаю фронт НЧ
	always @( posedge clk or negedge rst_n )
		if ( !rst_n )
			begin
				sh_clk_cam		 <= 1'h0;
				sh1_clk_cam      <= 1'h0;
			end
		else
			begin
				sh_clk_cam		 <= clk_cam;
				sh1_clk_cam      <= sh_clk_cam;			
			end

assign p_clk_cam	= sh_clk_cam & !sh1_clk_cam;	
			
	always @( posedge clk or negedge rst_n )
		if ( !rst_n )	
			cnt_href	<= 10'h0;
		else
			if ( p_VSYNC_cam )
				cnt_href	<= 10'h0;
			else if ( p_HREF_cam )
				cnt_href	<= cnt_href + 1'h1;			

//стробы в SDRAM
assign wr_strobe = ( ( glob_ns == s1_wr ) & ( glob_cs == s1_wait ) ) ? 1'h1: 1'h0;
assign rd_strobe = ( ( glob_ns == s1_rd ) & ( glob_cs == s1_wait ) ) ? 1'h1: 1'h0;

assign write_strobe = ( ( glob_ns == s1_wr ) || ( glob_cs == s1_wr )) ? 1'h1: 1'h0;
assign read_strobe = ( ( glob_ns == s1_rd ) || ( glob_cs == s1_rd )) ? 1'h1: 1'h0;
assign wait_strobe = ( ( glob_ns == s1_wait ) || ( glob_cs == s1_wait )) ? 1'h1: 1'h0;
assign idle_strobe = ( ( glob_ns == s1_idle ) || ( glob_cs == s1_idle )) ? 1'h1: 1'h0;

//строб чтения из входного ФИФО
	always @( posedge clk or negedge rst_n )
		if ( !rst_n )
			cnt_rd_input_fifo	<= 10'h0;
		else
			begin
				if ( cnt_rd_input_fifo == 10'd647 )
					cnt_rd_input_fifo	<= 10'h0;
				else if ( wr_strobe )
					cnt_rd_input_fifo	<= 10'h1;
				else if ( cnt_rd_input_fifo	!= 10'h0 )
					cnt_rd_input_fifo	<= cnt_rd_input_fifo + 1'd1;
			end
			
assign rd_input_fifo = ((( cnt_rd_input_fifo >= 10'd1)&(cnt_rd_input_fifo <= 10'd257))|(( cnt_rd_input_fifo >= 10'd261)&(cnt_rd_input_fifo <= 10'd516))|(cnt_rd_input_fifo >= 10'd520)) ? 1'h1 : 1'h0;		
	
endmodule	



