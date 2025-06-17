import React, { useRef } from "react";
import { Grid, Paper, Typography, Button } from "@mui/material";

const Displayer = ({ currentPair, handlePreference }) => {
  const bottomRef = useRef(null); // Ref to the bottom of the page

  const handleClick = (preference) => {
    handlePreference(preference); // Call the original preference handler
    if (bottomRef.current) {
      setTimeout(() => {
        bottomRef.current.scrollIntoView({ behavior: "smooth" });
      }, 500);
    }
  };

  return (
    <>
      <Grid container spacing={3} sx={{ marginBottom: "40px" }}>
        {[0, 1].map((index) => (
          <Grid item xs={12} sm={6} key={index}>
            <Paper
              elevation={3}
              style={{
                padding: "20px",
                backgroundColor:
                  currentPair.preference === `Vereinfachung ${index + 1}`
                    ? "#4caf50"
                    : "#666",
              }}
            >
              <Typography
                variant="body1"
                color="white"
                gutterBottom
                sx={{
                  textAlign: "left",
                  hyphens: "auto",
                  wordBreak: "break-word",
                  overflowWrap: "break-word",
                  wordSpacing: "0.1em",
                }}
              >
                Text {index + 1}:
                <br />
                <br />
                {currentPair.simplifications[index]}
                <br />
              </Typography>
              <Button
                variant="contained"
                color="success"
                onClick={() => handleClick(`Vereinfachung ${index + 1}`)}
              >
                Diesen Text verstehe ich besser
              </Button>
            </Paper>
          </Grid>
        ))}
      </Grid>
      {/* Invisible box at the bottom for scrolling */}
      <div ref={bottomRef} style={{ height: 1 }} />
    </>
  );
};

export default Displayer;
