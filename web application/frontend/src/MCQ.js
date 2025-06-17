import React from "react";
import { Grid, Paper, Typography, RadioGroup, FormControlLabel, FormControl, FormLabel, Radio } from "@mui/material";

const MCQ = ({ currentPair, multipleChoiceAnswer, handleMultipleChoiceChange }) => (
  currentPair.showMCQ && (
    <Grid item xs={12} sm={12}>
      <Paper
        elevation={3}
        style={{
          padding: "20px",
          marginBottom: "20px",
          marginTop: "20px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          backgroundColor: "#e5f6fd",
        }}
      >
        <FormControl component="fieldset" style={{ width: "100%" }}>
          <FormLabel
            component="legend"
            sx={{
              fontSize: "1.25rem",
            }}
          >
            Beantworten Sie die folgende Frage:
          </FormLabel>
          <Typography>{currentPair.question}</Typography>
          <RadioGroup
            name="multiple-choice"
            value={multipleChoiceAnswer || ""}
            onChange={handleMultipleChoiceChange}
          >
            {currentPair.options.map((option, index) => (
              <FormControlLabel
                key={index}
                value={String.fromCharCode(65 + index)}
                control={<Radio />}
                label={`${String.fromCharCode(65 + index)}. ${option}`}
              />
            ))}
          </RadioGroup>
        </FormControl>
      </Paper>
    </Grid>
  )
);

export default MCQ;
