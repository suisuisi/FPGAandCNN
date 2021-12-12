/** Simple frame-buffer based driver for the ILI9341 TFT module */
module tft_ili9341(
		input clk,
		input tft_sdo, output wire tft_sck, output wire tft_sdi, 
		output wire tft_dc, output reg tft_reset, output wire tft_cs,
		input[15:0] framebufferData, output wire framebufferClk
	);
	
	parameter INPUT_CLK_MHZ = 120; /* recommended */
	
	// Initial assignments
	initial tft_reset = 1'b1;

	// Assign pins and modules
	reg[8:0] spiData; 
	reg spiDataSet = 1'b0;
	wire spiIdle;
	
	reg frameBufferLowNibble = 1'b1;
	assign framebufferClk = !frameBufferLowNibble;
	
	tft_ili9341_spi spi(
		.spiClk(clk), 
		.data(spiData), .dataAvailable(spiDataSet),
		.tft_sck(tft_sck), .tft_sdi(tft_sdi), .tft_dc(tft_dc), .tft_cs(tft_cs),
		.idle(spiIdle));
	
	// Init Sequence Data (based upon https://github.com/notro/fbtft/blob/master/fb_ili9341.c)
	localparam INIT_SEQ_LEN = 64;
	reg[6:0] initSeqCounter = 7'b0;
	reg[8:0] INIT_SEQ [0:INIT_SEQ_LEN-1] = '{
		// Turn off Display
		{1'b0, 8'h28},
		// Init (??)
		{1'b0, 8'hCF}, {1'b1, 8'h00}, {1'b1, 8'h83}, {1'b1, 8'h30}, 
		{1'b0, 8'hED}, {1'b1, 8'h64}, {1'b1, 8'h03}, {1'b1, 8'h12}, {1'b1, 8'h81},
		{1'b0, 8'hE8}, {1'b1, 8'h85}, {1'b1, 8'h01}, {1'b1, 8'h79}, 
		{1'b0, 8'hCB}, {1'b1, 8'h39}, {1'b1, 8'h2C}, {1'b1, 8'h00}, {1'b1, 8'h34}, {1'b1, 8'h02},
		{1'b0, 8'hF7}, {1'b1, 8'h20},
		{1'b0, 8'hEA}, {1'b1, 8'h00}, {1'b1, 8'h00},
		// Power Control
		{1'b0, 8'hC0}, {1'b1, 8'h26},
		{1'b0, 8'hC1}, {1'b1, 8'h11},
		// VCOM
		{1'b0, 8'hC5}, {1'b1, 8'h35}, {1'b1, 8'h3E},
		{1'b0, 8'hC7}, {1'b1, 8'hBE},
		// Memory Access Control
		{1'b0, 8'h3A}, {1'b1, 8'h55},
		{1'b0, 8'h36}, {1'b1, 8'h20},
		{1'b0, 8'h2A}, {1'b1, 8'h00}, {1'b1, 8'h00}, {1'b1, 8'h01}, {1'b1, 8'h3F},
		{1'b0, 8'h2B}, {1'b1, 8'h00}, {1'b1, 8'h00}, {1'b1, 8'h00}, {1'b1, 8'hEF},
		// Frame Rate
		{1'b0, 8'hB1}, {1'b1, 8'h00}, {1'b1, 8'h1B},
		// Gamma
		{1'b0, 8'h26}, {1'b1, 8'h01},
		// Brightness
		{1'b0, 8'h51}, {1'b1, 8'hFF},
		// Display
		{1'b0, 8'hB7}, {1'b1, 8'h07},
		{1'b0, 8'hB6}, {1'b1, 8'h0A}, {1'b1, 8'h82}, {1'b1, 8'h27}, {1'b1, 8'h00},
		{1'b0, 8'h29}, // Enable Display
		{1'b0, 8'h2C} // Start  Memory-Write
	};
	
	
	// state machine with delay + idle support (used for initialization)
	reg[23:0] remainingDelayTicks = 24'b0;
	enum logic[2:0] { START, HOLD_RESET, WAIT_FOR_POWERUP, SEND_INIT_SEQ, LOOP} state = START;
	always @ (posedge clk) begin
		// clear data flag first
		spiDataSet <= 1'b0; 
		
		// always decrement delay ticks
		if (remainingDelayTicks > 0) begin
			remainingDelayTicks <= remainingDelayTicks - 1'b1;
		end
		else if (spiIdle && !spiDataSet) begin
			// advance state machine to next state, but only do this if we
			// didn't just clock in the last byte (since idle is not yet updated)
			case (state)
				// initialize all pins in START mode; reset the LCD
				START: begin
					tft_reset <= 1'b0;
					remainingDelayTicks <= 24'(INPUT_CLK_MHZ * 10); // min: 10us
					state <= HOLD_RESET;
				end
				
				// wait for RESET to kick in; then release pin & wait for power up
				HOLD_RESET: begin
					tft_reset <= 1'b1; // release pin
					remainingDelayTicks <= 24'(INPUT_CLK_MHZ * 120000); // min: 120ms
					state <= WAIT_FOR_POWERUP;
					frameBufferLowNibble <= 1'b0; // request first pixel
				end
				
				// if power up is completed -> sw reset
				WAIT_FOR_POWERUP: begin
					spiData <= {1'b0, 8'h11}; // take out of sleep mode
					spiDataSet <= 1'b1;
					remainingDelayTicks <= 24'(INPUT_CLK_MHZ * 5000); // min: 5ms
					state <= SEND_INIT_SEQ;
					frameBufferLowNibble <= 1'b1;
				end
				
				// setup the LCD by sending the init sequence
				SEND_INIT_SEQ: begin
					if (initSeqCounter < INIT_SEQ_LEN) begin
						spiData <= INIT_SEQ[initSeqCounter];
						spiDataSet <= 1'b1;
						initSeqCounter <= initSeqCounter + 1'b1;
					end else begin
						state <= LOOP;
						remainingDelayTicks <= 24'(INPUT_CLK_MHZ * 10000); // min: 10ms
					end
				end
				
				// frame buffer loop
				default: begin
					spiData <= !frameBufferLowNibble ? {1'b1, framebufferData[15:8]} :{1'b1, framebufferData[7:0]};
					spiDataSet <= 1'b1;
					frameBufferLowNibble <= !frameBufferLowNibble;
				end
			endcase
		end
	end
endmodule