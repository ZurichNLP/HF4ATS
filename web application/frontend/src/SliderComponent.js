import React from "react";
import { Grid, Paper, Typography, Slider } from "@mui/material";
import { sliderConfigs } from "./sliderConfigs";

function valuetext(value) {
  return `${value}%`;
}

const SliderComponent = ({ currentPair, handleSliderChange }) =>
  sliderConfigs.map((config, index) => (
    <Grid item xs={12} sm={12} key={config.name}>
      <Paper
        elevation={3}
        style={{
          padding: "20px",
          marginBottom: "20px",
          marginTop: index === 0 ? "30px" : "20px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          backgroundColor: "#666",
        }}
      >
        <Typography color="white" style={{ alignSelf: "flex-start" }}>
          {config.question}
        </Typography>
        <Slider
          aria-label={config.name}
          value={currentPair.sliders[config.name]}
          onChange={handleSliderChange(config.name)}
          getAriaValueText={valuetext}
          step={1}
          min={1}
          max={5}
          valueLabelDisplay="off"
          marks={config.marks}
          sx={{
            width: "90%",
            margin: "20px auto 0",
            marginBottom: "70px",
            color: "white",
            "& .MuiSlider-markLabel": {
              color: "white",
              textAlign: "center",
              whiteSpace: "nowrap",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            },
          }}
        />
      </Paper>
    </Grid>
  ));

export default SliderComponent;
