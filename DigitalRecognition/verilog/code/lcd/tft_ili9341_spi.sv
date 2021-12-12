// --- Byte-wise SPI + DC implementation
// * Will copy data into internal buffer
// * 'Idle' will be set to 0 once buffer copy is complete
// * Data is only copied if 'dataAvailable' is set to 1
// * SPI CLK will stop (high state) if no data is being sent
module tft_ili9341_spi(
		input spiClk, input[8:0] data, input dataAvailable,
		output wire tft_sck, output reg tft_sdi, output reg tft_dc, output wire tft_cs,
		output reg idle
	);

	// Registers
	reg[0:2] counter = 3'b0;
	reg[8:0] internalData;
	reg internalSck;
	reg cs;
	
	initial internalSck <= 1'b1;
	initial idle <= 1'b1;
	initial cs <= 1'b0;
	
	// Combinational Assignments
	wire dataDc = internalData[8];
	wire[0:7] dataShift = internalData[7:0]; // MSB first
	
	assign tft_sck = internalSck & cs; // only drive sck with an active CS
	assign tft_cs = !cs; // active low
	
	// Update SPI CLK + Output data
	always @ (posedge spiClk) begin
		// Store new data in internal register
		if (dataAvailable) begin
			internalData <= data;
			idle <= 1'b0;
		end
		
		// Change data if we're actively sending
		if (!idle) begin
			// Toggle Clock on every active tick
			internalSck <= !internalSck;
				
			// Check if SCK will be low next
			if (internalSck) begin
				// Update pins
				tft_dc <= dataDc;
				tft_sdi <= dataShift[counter];
				cs <= 1'b1;
				
				// Advance counter
				counter <= counter + 1'b1;
				idle <= &counter; // we're just sending the last bit
			end
		end
		else begin
			internalSck <= 1'b1; // idle mode (also: sent last bit)
			if (internalSck) cs <= 1'b0; // idle for two bits in a row -> deactivate CS
		end
	end	
endmodule