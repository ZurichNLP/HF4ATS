import { createContext, useState, useContext } from "react";

const SharedContext = createContext();

export const SharedContextProvider = ({ children }) => {
  const [sharedState, setSharedState] = useState("");

  return (
    <SharedContext.Provider value={{ sharedState, setSharedState }}>
      {children}
    </SharedContext.Provider>
  );
};

export const useSharedContext = () => useContext(SharedContext);
