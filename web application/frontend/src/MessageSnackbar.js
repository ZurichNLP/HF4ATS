import React from "react";
import { Snackbar, Alert } from "@mui/material";

const CustomSnackbar = ({ successMessageOpen, handleSnackbarClose }) => (
  <Snackbar
    open={successMessageOpen}
    autoHideDuration={6000}
    onClose={handleSnackbarClose}
    anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
  >
    <Alert
      onClose={handleSnackbarClose}
      severity="success"
      sx={{ width: "100%" }}
    >
      Wir haben Ihre Antworten empfangen. Vielen Dank!
    </Alert>
  </Snackbar>
);

export default CustomSnackbar;
