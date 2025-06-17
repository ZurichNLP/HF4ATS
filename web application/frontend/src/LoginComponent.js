import React, { useState } from "react";
import { TextField, Button, Container, Box, Typography } from "@mui/material";
import { styled } from "@mui/system";

const verifyLoginName = (loginName) => {
  if (loginName.length !== 4) {
    return false;
  }

  const lowerInput = loginName.toLowerCase();

  const validPrefixes = ["ta", "ea", "te", "ee", "us"];
  const validSuffixes = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
  ];

  const prefix = lowerInput.slice(0, 2);
  const suffix = lowerInput.slice(2);

  return validPrefixes.includes(prefix) && validSuffixes.includes(suffix);
};

const CustomTextField = styled(TextField)({
  "& .MuiOutlinedInput-root": {
    backgroundColor: "#666",
    "& fieldset": {
      borderColor: "white",
    },
    "&:hover fieldset": {
      borderColor: "white",
    },
    "&.Mui-focused fieldset": {
      borderColor: "white",
    },
  },
  "& .MuiInputLabel-root": {
    color: "white",
    fontSize: "1rem",
  },

  "& .MuiInputLabel-root.Mui-focused": {
    color: "white",
    fontSize: "1rem",
  },
});

function LoginComponent({ onLogin, onUserIDChange }) {
  const [inputValue, setInputValue] = useState("");
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  const handleLoginClick = () => {
    const loginName = inputValue;

    if (verifyLoginName(loginName)) {
      console.log("Login successful");
      setIsLoggedIn(true);
      onLogin();
    } else {
      const alertMessage = "Fehler! Bitte überprüfen Sie Ihre User-ID.";
      alert(alertMessage);
    }
  };

  const handleTextFieldChange = (event) => {
    setInputValue(event.target.value);
    onUserIDChange(event.target.value);
  };

  return (
    <Container maxWidth="sm">
      <Box
        display="flex"
        flexDirection="column"
        alignItems="center"
        mt={10}
        sx={{ backgroundColor: "#666", padding: 2 }}
      >
        {!isLoggedIn && (
          <>
            <CustomTextField
              onChange={handleTextFieldChange}
              label="Bitte geben Sie Ihre User-ID ein ..."
              variant="outlined"
              value={inputValue}
              fullWidth
              margin="normal"
              sx={{ width: "80%" }}
            />
            <Button
              variant="contained"
              color="primary"
              onClick={handleLoginClick}
            >
              Einloggen
            </Button>
          </>
        )}
        {isLoggedIn && (
          <Box my={4} textAlign="center">
            <Typography
              value={inputValue}
              variant="h5"
              component="h1"
              color="white"
              sx={{
                backgroundColor: "#666",
                // padding: "5px",
                borderRadius: "4px",
                display: "inline-block",
              }}
            >
              Sie sind eingeloggt: {inputValue.toLowerCase()}
            </Typography>
          </Box>
        )}
      </Box>
    </Container>
  );
}

export default LoginComponent;
