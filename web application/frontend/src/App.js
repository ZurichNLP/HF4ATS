import React, { useState, useEffect, useRef } from "react";
import { Box, Container, Typography } from "@mui/material";
import Dataloader from "./Dataloader";
import LoginComponent from "./LoginComponent.js";
import Original from "./Original.js";
import Simplifications from "./Simplifications.js";
// import SliderComponent from "./SliderComponent";
import Submission from "./Submission";
import MessageSnackbar from "./MessageSnackbar";
import uzhLogo from "./uzh_on_blue.png";
import zhawLogo from "./zhaw_blue.png";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import Papa from "papaparse";
import "./App.css";
// import { TimerOutlined } from "@mui/icons-material";
// import { Block } from "@mui/icons-material";

const theme = createTheme({
  palette: {
    primary: {
      main: "#1976d2",
    },
    secondary: {
      main: "#dc004e",
    },
    background: {
      default: "#f0f0f0",
    },
  },
});

const App = () => {
  const [textPairs, setTextPairs] = useState([]);
  const [currentPairIndex, setCurrentPairIndex] = useState(0);
  const [isDatasetLoaded, setIsDatasetLoaded] = useState(false);
  const [successMessageOpen, setSuccessMessageOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userID, setUserID] = useState("");

  const bottomRef = useRef(null); // Ref for the bottom of the page

  useEffect(() => {
    if (isLoggedIn && bottomRef.current) {
      // Use setTimeout to ensure all content is loaded before scrolling
      setTimeout(() => {
        bottomRef.current.scrollIntoView({ behavior: "smooth" });
      }, 500);
    }
  }, [isLoggedIn]);

  const handleLogin = () => {
    setIsLoggedIn(true);
  };

  const handleUserIDChange = (id) => {
    setUserID(id);
  };

  const handleSubmit = () => {
    if (textPairs.length === 0 || !currentPair.preference) {
      return;
    }

    const jsonlContent = textPairs
      .map((pair) => JSON.stringify(pair))
      .join("\n");

    fetch(
      // Deployment on UZH Science Cluster, consult Yingqiang for the server address
      "https://pub.cl.uzh.ch/projects/dpo4ats/api/save-json/",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(jsonlContent),
      }
    )
      .then(() => {
        console.log("Request sent.");
        setSuccessMessageOpen(true);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  const handleSnackbarClose = () => {
    setSuccessMessageOpen(false);
  };

  const loadCSV = (fileName) => {
    fetch(fileName)
      .then((response) => response.text())
      .then((data) => {
        Papa.parse(data, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            const parsedTextPairs = results.data.map((row) => ({
              userID: userID,
              datasetID: fileName.slice(-5, -4), // Dataset ID is always the fifth character counting backwards from the file name
              phaseID: userID[1].toLowerCase() === "a" ? "anno" : "eval",
              original: row.original,
              simplifications: [row.simplification1, row.simplification2],
              question: row.question,
              options: row.options ? row.options.split(" | ") : [],
              answer: row.answer,
              preference: null,
              multipleChoiceAnswer: null,
              sliders: {
                geometry: 3,
                lexicon: 3,
                semantics: 3,
                syntax: 3,
              },
            }));
            setTextPairs(parsedTextPairs);
            setCurrentPairIndex(0);
            setIsDatasetLoaded(true);
          },
          error: (error) => {
            console.error("Error parsing CSV file:", error.message);
          },
        });
      })
      .catch((error) => console.error("Error loading CSV file:", error));
  };

  const handlePreference = (choice) => {
    const updatedTextPairs = [...textPairs];
    updatedTextPairs[currentPairIndex].preference = choice;
    setTextPairs(updatedTextPairs);
  };

  // const handleSliderChange = (sliderName) => (event, newValue) => {
  // const updatedTextPairs = [...textPairs];
  // updatedTextPairs[currentPairIndex].sliders[sliderName] = newValue;
  // setTextPairs(updatedTextPairs);
  // };

  const handleNext = () => {
    if (currentPairIndex < textPairs.length - 1) {
      setCurrentPairIndex(currentPairIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (currentPairIndex > 0) {
      setCurrentPairIndex(currentPairIndex - 1);
    }
  };

  // const handleShowTS = () => {
  //   const updatedTextPairs = [...textPairs];
  //   updatedTextPairs[currentPairIndex].showTS = true;
  //   setTextPairs(updatedTextPairs);
  // };

  const currentPair = textPairs[currentPairIndex] || {};
  const progress =
    textPairs.length > 0
      ? ((currentPairIndex + 1) / textPairs.length) * 100
      : 0;

  return (
    <ThemeProvider theme={theme}>
      {/* <div className="video-background">
        <video autoPlay loop muted className="video-background__content">
          <source
            src={`${process.env.PUBLIC_URL}/snow-carbin.mp4`}
            type="video/mp4"
          />
        </video>
      </div> */}
      <div
        style={{ minHeight: "130vh", display: "flex", flexDirection: "column" }}
      >
        <Container maxWidth="md" sx={{ height: "auto", overflow: "auto" }}>
          <Box
            display="flex"
            alignItems="center"
            justifyContent="center"
            my={4}
            sx={{
              width: "fit-content",
              padding: 2,
              border: "2px solid white",
              backgroundColor: "#666",
              margin: "0 auto",
            }}
          >
            <img
              src={uzhLogo}
              alt="University Logo"
              style={{ height: 60, marginRight: 20 }}
            />
            <img
              src={zhawLogo}
              alt="University Logo"
              style={{ height: 60, marginRight: 5 }}
            />
            <Box sx={{ ml: 2, display: "flex", alignItems: "center" }}>
              <Box sx={{ mr: 3 }}>
                <Typography color="white" sx={{ textAlign: "left" }}>
                  Universität Zürich
                </Typography>
                <Typography color="white" sx={{ textAlign: "left" }}>
                  Department für Computerlinguistik
                </Typography>
              </Box>
              <Box>
                <Typography color="white" sx={{ textAlign: "left" }}>
                  Zürcher Hochschule für Angewandte Wissenschaften
                </Typography>
                <Typography color="white" sx={{ textAlign: "left" }}>
                  Department für Angewandte Linguistik
                </Typography>
              </Box>
            </Box>
          </Box>

          <LoginComponent
            onLogin={handleLogin}
            onUserIDChange={handleUserIDChange}
          />

          {isLoggedIn && (
            <Dataloader
              userID={userID}
              loadCSV={loadCSV}
              isDatasetLoaded={isDatasetLoaded}
              progress={progress}
              currentPairIndex={currentPairIndex}
              textPairs={textPairs}
              handleNext={handleNext}
              handlePrevious={handlePrevious}
            />
          )}

          {textPairs.length > 0 && currentPair && (
            <>
              <Box textAlign="center" mt={4}>

              <Submission
                  currentPairIndex={currentPairIndex}
                  textPairs={textPairs}
                  currentPair={currentPair}
                  handleSubmit={handleSubmit}
                />
              
              {(userID === "ea02" || userID === "ea04" || userID === "ea06" || userID === "ea08" || userID === "ea10" || userID === "ea12" || userID === "ea14" || userID === "ea16" || userID === "ea18" || userID === "ea20") && (
                <Box textAlign="center" mt={4}>
                  <Original currentPair={currentPair} />
                </Box>
              )}

                {/* <Original currentPair={currentPair} handleShowTS={handleShowTS} /> */}
                {/* {currentPair.showTS && (
                <Simplifications
                  currentPair={currentPair}
                  handlePreference={handlePreference}
                />
              )} */}


                <Simplifications
                  currentPair={currentPair}
                  handlePreference={handlePreference}
                />

                {/*{currentPair.preference && (
                <SliderComponent
                  currentPair={currentPair}
                  handleSliderChange={handleSliderChange}
                />
              )}*/}
              </Box>
            </>
          )}

          {/* Invisible box at the bottom for scrolling */}
          <Box ref={bottomRef} sx={{ height: 1 }} />
        </Container>
      </div>

      <MessageSnackbar
        successMessageOpen={successMessageOpen}
        handleSnackbarClose={handleSnackbarClose}
      />
    </ThemeProvider>
  );
};

export default App;
