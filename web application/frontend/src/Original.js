import React from "react";
import { Grid, Paper, Typography, Button, Box } from "@mui/material";

/* const Original = ({ currentPair, handleShowTS }) => (
  <>
    <Box alignItems="center" justifyContent="center">
      <Grid item xs={12} sm={12} my={5}>
        <Paper
          elevation={3}
          style={{
            padding: "20px",
            marginBottom: "20px",
            backgroundColor: "#666",
          }}
        >
          <Typography color="white" sx={{ textAlign: "justify" }}>
            Originaler Text:
            <br />
            <br />
            {currentPair.original}
          </Typography>
          <div
            style={{
              display: "flex",
              justifyContent: "flex-end",
              marginTop: "20px",
            }}
          >
            <Button variant="contained" color="primary" onClick={handleShowTS}>
              Gelesen
            </Button>
          </div>
        </Paper>
      </Grid>
    </Box>
  </>
);

export default Original; */

const Original = ({ currentPair }) => (
  <Box alignItems="center" justifyContent="center">
    <Grid item xs={12} sm={12} my={5}>
      <Paper
        elevation={3}
        style={{
          padding: "20px",
          marginBottom: "20px",
          backgroundColor: "#666",
        }}
      >
        <Typography color="white" sx={{ textAlign: "justify" }}>
          Originaler Text:
          <br />
          <br />
          {currentPair.original}
        </Typography>
      </Paper>
    </Grid>
  </Box>
);

export default Original;