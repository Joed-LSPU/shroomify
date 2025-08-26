'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
  id: number;
  email: string;
  full_name: string;
  created_at: string;
}

interface AuthContextType {
  isLoggedIn: boolean;
  user: User | null;
  login: (userData: User) => void;
  logout: () => void;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  // Load user data from localStorage on app start
  useEffect(() => {
    const savedUser = localStorage.getItem('shroomify_user');
    const savedLoginState = localStorage.getItem('shroomify_logged_in');
    
    if (savedUser && savedLoginState === 'true') {
      try {
        const userData = JSON.parse(savedUser);
        setUser(userData);
        setIsLoggedIn(true);
      } catch (error) {
        console.error('Error parsing saved user data:', error);
        // Clear invalid data
        localStorage.removeItem('shroomify_user');
        localStorage.removeItem('shroomify_logged_in');
      }
    }
    
    setLoading(false);
  }, []);

  const login = (userData: User) => {
    setUser(userData);
    setIsLoggedIn(true);
    // Save to localStorage
    localStorage.setItem('shroomify_user', JSON.stringify(userData));
    localStorage.setItem('shroomify_logged_in', 'true');
  };

  const logout = () => {
    setUser(null);
    setIsLoggedIn(false);
    // Clear from localStorage
    localStorage.removeItem('shroomify_user');
    localStorage.removeItem('shroomify_logged_in');
  };

  const value = {
    isLoggedIn,
    user,
    login,
    logout,
    loading
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
