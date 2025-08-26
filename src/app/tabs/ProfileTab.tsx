'use client';
import { User, Settings, Camera, Calendar, Edit3, Mail, Phone, Trophy, Target, Eye, EyeOff, Loader2 } from 'lucide-react';
import React, { JSX, useState, useEffect } from 'react';
import { supabase } from '../../lib/supabaseClient';
import { useAuth } from '../../lib/AuthContext';

// Type definitions
interface UserData {
  name: string;
  email: string;
  phone: string;
  joinDate: string;
  experienceLevel: string;
  favoriteMethod: string;
  avatar?: string | null;
}

interface ActivityItem {
  id: number;
  type: 'scan' | 'achievement' | 'alert';
  title: string;
  time: string;
  status: string;
  statusColor: 'green' | 'yellow' | 'red' | 'blue';
}

interface ProfileTabProps {
  userData?: UserData;
  onUpdateProfile?: (data: UserData) => void;
  onUpdateAvatar?: () => void;
  activityData?: ActivityItem[];
  personalizedTip?: string;
  isLoggedIn?: boolean;
  onLogin?: (email: string, password: string) => Promise<void>;
  onSignUp?: (email: string, password: string, name: string) => Promise<void>;
  onGoogleLogin?: () => Promise<void>;
  authLoading?: boolean;
  authError?: string | null;
}

const ProfileTab: React.FC<ProfileTabProps> = ({ 
  userData = {
    name: "Juan Dela Cruz",
    email: "juan.delacruz@shroomify.com",
    phone: "+63 912 345 6789",
    joinDate: "January 2024",
    experienceLevel: "Intermediate",
    favoriteMethod: "Straw Substrate",
    avatar: null
  },
  onUpdateProfile,
  onUpdateAvatar,
  activityData = [
    {
      id: 1,
      type: "scan",
      title: "Scanned fruiting bag #23",
      time: "2 hours ago",
      status: "Healthy",
      statusColor: "green"
    },
    {
      id: 2,
      type: "achievement",
      title: "Achieved 12-day streak",
      time: "1 day ago",
      status: "Achievement",
      statusColor: "yellow"
    },
    {
      id: 3,
      type: "alert",
      title: "Detected contamination early",
      time: "3 days ago",
      status: "Alert",
      statusColor: "red"
    }
  ],
  personalizedTip = "Based on your cultivation history, consider trying the hardwood sawdust method next! Your success rate with straw substrate shows you're ready for intermediate techniques.",
  isLoggedIn = false,
  onLogin,
  onSignUp,
  onGoogleLogin,
  authLoading: propAuthLoading = false,
  authError = null
}) => {
  const { isLoggedIn: globalIsLoggedIn, user: globalUser, login: globalLogin, logout: globalLogout, loading: globalAuthLoading } = useAuth();
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [localUserData, setLocalUserData] = useState<UserData>(
    globalUser ? {
      name: globalUser.full_name,
      email: globalUser.email,
      phone: "+63 912 345 6789", // Default phone
      joinDate: new Date(globalUser.created_at).toLocaleDateString('en-US', { month: 'long', year: 'numeric' }),
      experienceLevel: "Intermediate",
      favoriteMethod: "Straw Substrate",
      avatar: null
    } : userData
  );
  
  // Auth form states
  const [isSignUp, setIsSignUp] = useState<boolean>(false);
  const [email, setEmail] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [name, setName] = useState<string>('');
  const [showPassword, setShowPassword] = useState<boolean>(false);
  const [formError, setFormError] = useState<string>('');
  const [formSuccess, setFormSuccess] = useState<string>('');
  const [localAuthLoading, setLocalAuthLoading] = useState<boolean>(false);

  // Update local user data when global user changes
  useEffect(() => {
    if (globalUser) {
      setLocalUserData({
        name: globalUser.full_name,
        email: globalUser.email,
        phone: "+63 912 345 6789", // Default phone
        joinDate: new Date(globalUser.created_at).toLocaleDateString('en-US', { month: 'long', year: 'numeric' }),
        experienceLevel: "Intermediate",
        favoriteMethod: "Straw Substrate",
        avatar: null
      });
    }
  }, [globalUser]);

  // Handle profile updates
  const handleProfileUpdate = (field: keyof UserData, value: string): void => {
    const updatedData = { ...localUserData, [field]: value };
    setLocalUserData(updatedData);
    if (onUpdateProfile) {
      onUpdateProfile(updatedData);
    }
  };

  // Handle avatar update
  const handleAvatarUpdate = (): void => {
    if (onUpdateAvatar) {
      onUpdateAvatar();
    }
  };

  // Generate initials from name
  const getInitials = (name: string): string => {
    return name.split(' ').map((n: string) => n[0]).join('').toUpperCase();
  };

  // Get activity icon based on type
  const getActivityIcon = (type: ActivityItem['type']): JSX.Element => {
    const iconMap = {
      scan: Camera,
      achievement: Trophy,
      alert: Target
    };
    const IconComponent = iconMap[type];
    return <IconComponent className="w-4 h-4" />;
  };

  // Get status color classes
  const getStatusColorClasses = (color: ActivityItem['statusColor']) => {
    const colorMap = {
      green: {
        bg: "bg-green-600/20",
        text: "text-green-400",
        iconBg: "bg-green-600/20"
      },
      yellow: {
        bg: "bg-yellow-600/20",
        text: "text-yellow-400",
        iconBg: "bg-yellow-600/20"
      },
      red: {
        bg: "bg-red-600/20",
        text: "text-red-400",
        iconBg: "bg-red-600/20"
      },
      blue: {
        bg: "bg-blue-600/20",
        text: "text-blue-400",
        iconBg: "bg-blue-600/20"
      }
    };
    return colorMap[color];
  };

  // Save user to database
  const saveUserToDatabase = async (userEmail: string, userPassword: string, userName: string) => {
    try {
      const { data, error } = await supabase
        .from('Users')
        .insert([
          {
            email: userEmail,
            password: userPassword,
            full_name: userName,
            created_at: new Date().toISOString().split('T')[0] // Format as YYYY-MM-DD
          }
        ])
        .select();

      if (error) {
        console.error('Database error:', error);
        throw new Error(error.message);
      }

      console.log('User saved successfully:', data);
      return data;
    } catch (error) {
      console.error('Error saving user:', error);
      throw error;
    }
  };

  // Check user credentials for login
  const checkUserCredentials = async (userEmail: string, userPassword: string) => {
    try {
      const { data, error } = await supabase
        .from('Users')
        .select('*')
        .eq('email', userEmail)
        .eq('password', userPassword)
        .single();

      if (error) {
        console.error('Database error:', error);
        throw new Error(error.message);
      }

      if (!data) {
        throw new Error('Invalid email or password');
      }

      console.log('User logged in successfully:', data);
      return data;
    } catch (error) {
      console.error('Error checking credentials:', error);
      throw error;
    }
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError('');
    setFormSuccess('');

    // Enhanced validation
    if (!email.trim()) {
      setFormError('Please enter your email address');
      return;
    }

    if (!password.trim()) {
      setFormError('Please enter your password');
      return;
    }

    if (password.length < 6) {
      setFormError('Password must be at least 6 characters long');
      return;
    }

    if (isSignUp && !name.trim()) {
      setFormError('Please enter your full name');
      return;
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setFormError('Please enter a valid email address');
      return;
    }

    try {
      setLocalAuthLoading(true);
      
      if (isSignUp) {
        // Save to database first
        await saveUserToDatabase(email.trim(), password, name.trim());
        
        // Show success message
        setFormSuccess('Account created successfully!');
        
        // Clear form
        setEmail('');
        setPassword('');
        setName('');
        
        // Then call the original onSignUp if it exists
        if (onSignUp) {
          await onSignUp(email.trim(), password, name.trim());
        }
      } else {
        // Login logic
        const userData = await checkUserCredentials(email.trim(), password);
        
        // Show success message
        setFormSuccess('Login successful! Welcome back! 🍄');
        
        // Clear form
        setEmail('');
        setPassword('');
        
        // Use global login function
        globalLogin(userData);
        
        // Call the original onLogin if it exists
        if (onLogin) {
          await onLogin(email.trim(), password);
        }
      }
    } catch (error) {
      console.error('Authentication error:', error);
      setFormError('Authentication failed. Please try again.');
    } finally {
      setLocalAuthLoading(false);
    }
  };

  // Handle Google login
  const handleGoogleLogin = async () => {
    if (onGoogleLogin) {
      try {
        await onGoogleLogin();
      } catch (error) {
        setFormError('Google login failed. Please try again.');
      }
    }
  };

  // Reset form when switching between login/signup
  const switchMode = () => {
    setIsSignUp(!isSignUp);
    setFormError('');
    setFormSuccess('');
    setEmail('');
    setPassword('');
    setName('');
  };

  // Handle logout
  const handleLogout = () => {
    globalLogout();
    setFormError('');
    setFormSuccess('');
  };

  return (
    <div className="p-6 space-y-6 relative">
      {/* Profile Card */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        <div className="relative bg-gradient-to-br from-green-600/20 to-blue-600/20 p-6">
          <div className="absolute top-4 right-4">
            <button 
              onClick={() => setIsEditing(!isEditing)}
              className="bg-gray-700/50 hover:bg-gray-600/50 p-2 rounded-lg transition-colors"
              aria-label="Edit profile"
            >
              <Edit3 className="w-4 h-4 text-gray-300" />
            </button>
          </div>
          <div className="flex items-center space-x-4">
            <div className="relative">
              {localUserData.avatar ? (
                <img 
                  src={localUserData.avatar} 
                  alt="Profile" 
                  className="w-20 h-20 rounded-full object-cover"
                />
              ) : (
                <div className="w-20 h-20 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center text-white text-2xl font-bold">
                  {getInitials(localUserData.name)}
                </div>
              )}
              <button 
                onClick={handleAvatarUpdate}
                className="absolute -bottom-1 -right-1 bg-gray-700 hover:bg-gray-600 p-1.5 rounded-full transition-colors border-2 border-gray-800"
                aria-label="Update avatar"
              >
                <Camera className="w-3 h-3 text-gray-300" />
              </button>
            </div>
            <div className="flex-1">
              {isEditing ? (
                <input
                  type="text"
                  value={localUserData.name}
                  onChange={(e) => handleProfileUpdate('name', e.target.value)}
                  className="text-xl font-semibold text-white bg-transparent border-b border-gray-600 focus:border-green-400 outline-none mb-1"
                  onBlur={() => setIsEditing(false)}
                  autoFocus
                />
              ) : (
                <h3 className="text-xl font-semibold text-white mb-1">{localUserData.name}</h3>
              )}
              <div className="flex items-center text-green-400 text-sm mb-2">
                <span className="w-2 h-2 bg-green-400 rounded-full mr-2"></span>
                <span>{localUserData.experienceLevel} Cultivator</span>
              </div>
              <div className="flex items-center text-gray-400 text-sm">
                <Calendar className="w-4 h-4 mr-1" />
                <span>Member since {localUserData.joinDate}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Contact Info */}
        <div className="p-4 border-b border-gray-700">
          <div className="space-y-2">
            <div className="flex items-center text-gray-300 text-sm">
              <Mail className="w-4 h-4 mr-3 text-gray-400" />
              <span>{localUserData.email}</span>
            </div>
            <div className="flex items-center text-gray-300 text-sm">
              <Phone className="w-4 h-4 mr-3 text-gray-400" />
              <span>{localUserData.phone}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Cultivation Preferences */}
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-3">🍄 Cultivation Preferences</h3>
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-300">Favorite Method:</span>
            <span className="text-green-400 font-medium">{localUserData.favoriteMethod}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-300">Experience Level:</span>
            <span className="bg-yellow-600/20 text-yellow-400 px-2 py-1 rounded text-sm">
              {localUserData.experienceLevel}
            </span>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-3">📊 Recent Activity</h3>
        <div className="space-y-3">
          {activityData.map((activity: ActivityItem) => {
            const statusColors = getStatusColorClasses(activity.statusColor);
            return (
              <div key={activity.id} className="flex items-center space-x-3">
                <div className={`w-8 h-8 ${statusColors.iconBg} rounded-full flex items-center justify-center`}>
                  <span className={statusColors.text}>
                    {getActivityIcon(activity.type)}
                  </span>
                </div>
                <div className="flex-1">
                  <p className="text-gray-300 text-sm">{activity.title}</p>
                  <p className="text-gray-500 text-xs">{activity.time}</p>
                </div>
                <span className={`${statusColors.bg} ${statusColors.text} px-2 py-1 rounded text-xs`}>
                  {activity.status}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Tips for You */}
      <div className="bg-gradient-to-r from-green-600/10 to-blue-600/10 rounded-lg p-4 border border-green-600/20">
        <h3 className="text-lg font-semibold text-white mb-2">💡 Personalized Tip</h3>
        <p className="text-gray-300 text-sm">
          {personalizedTip}
        </p>
      </div>

      {/* Settings and Logout Buttons */}
      <div className="text-center space-y-3">
        <button className="bg-gradient-to-r from-gray-700 to-gray-600 hover:from-gray-600 hover:to-gray-500 text-white font-semibold py-3 px-8 rounded-lg shadow-lg transform hover:scale-105 transition-all duration-200 flex items-center space-x-2 mx-auto">
          <Settings className="w-5 h-5" />
          <span>Account Settings</span>
        </button>
        
        <button 
          onClick={handleLogout}
          className="bg-gradient-to-r from-red-600 to-red-500 hover:from-red-500 hover:to-red-400 text-white font-semibold py-3 px-8 rounded-lg shadow-lg transform hover:scale-105 transition-all duration-200 flex items-center space-x-2 mx-auto"
        >
          <span>Logout</span>
        </button>
      </div>

      {/* Login Gate Overlay */}
      {!globalIsLoggedIn && (
        <>
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-20" />
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 overflow-y-auto">
            <div className="bg-gray-800 rounded-xl border border-gray-700 shadow-2xl max-w-md w-full overflow-hidden my-8">
                              <div className="bg-gradient-to-br from-green-600/20 to-blue-600/20 p-4 text-center">
                <div className="mx-auto mb-3 w-14 h-14 rounded-full bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center">
                  <User className="w-7 h-7 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-white">
                  {isSignUp ? 'Join The KabuTeam' : 'Welcome Back'}
                </h3>
                <p className="text-gray-300 text-xs mt-1">
                  {isSignUp 
                    ? 'Create your account to start your cultivation journey' 
                    : 'Sign in to access your profile and cultivation data'
                  }
                </p>
              </div>
              
              <form onSubmit={handleSubmit} className="p-4 max-h-[50vh] overflow-y-auto scrollbar-hide">
                <div className="space-y-3">
                  {/* Name field for signup */}
                  {isSignUp && (
                    <div>
                      <label htmlFor="name" className="block text-sm font-medium text-gray-300 mb-2">
                        Full Name
                      </label>
                      <input
                        id="name"
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-green-400 focus:ring-1 focus:ring-green-400"
                        placeholder="Enter your full name"
                        required={isSignUp}
                        autoComplete="name"
                        aria-describedby={isSignUp ? "name-error" : undefined}
                      />
                    </div>
                  )}

                  {/* Email field */}
                  <div>
                    <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
                      Email Address
                    </label>
                    <input
                      id="email"
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-green-400 focus:ring-1 focus:ring-green-400"
                      placeholder="Enter your email"
                      required
                      autoComplete="email"
                      aria-describedby="email-error"
                    />
                  </div>

                  {/* Password field */}
                  <div>
                    <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
                      Password
                    </label>
                    <div className="relative">
                      <input
                        id="password"
                        type={showPassword ? "text" : "password"}
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-green-400 focus:ring-1 focus:ring-green-400 pr-10"
                        placeholder="Enter your password"
                        required
                        minLength={6}
                        autoComplete={isSignUp ? "new-password" : "current-password"}
                        aria-describedby="password-error"
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-300"
                        aria-label={showPassword ? "Hide password" : "Show password"}
                      >
                        {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                  </div>

                  {/* Success display */}
                  {formSuccess && (
                    <div className="bg-green-600/10 border border-green-600/20 rounded-lg p-3">
                      <p className="text-green-400 text-sm">{formSuccess}</p>
                    </div>
                  )}

                  {/* Error display */}
                  {(formError || authError) && (
                    <div className="bg-red-600/10 border border-red-600/20 rounded-lg p-3">
                      <p className="text-red-400 text-sm">{formError || authError}</p>
                    </div>
                  )}

                  {/* Submit button */}
                  <button
                    type="submit"
                    disabled={localAuthLoading || globalAuthLoading}
                    className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 disabled:from-gray-600 disabled:to-gray-600 text-white font-semibold py-3 px-4 rounded-lg transition-all flex items-center justify-center space-x-2"
                  >
                    {localAuthLoading || globalAuthLoading ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <span>{isSignUp ? 'Create Account' : 'Sign In'}</span>
                    )}
                  </button>
                </div>

                {/* Divider */}
                <div className="flex items-center my-3">
                  <div className="flex-1 border-t border-gray-600"></div>
                  <span className="px-3 text-sm text-gray-400">or</span>
                  <div className="flex-1 border-t border-gray-600"></div>
                </div>

                {/* Google login button */}
                <button
                  onClick={handleGoogleLogin}
                  disabled={localAuthLoading || globalAuthLoading}
                  className="w-full bg-white hover:bg-gray-100 disabled:bg-gray-300 text-gray-900 font-semibold py-3 px-4 rounded-lg transition-all flex items-center justify-center space-x-2 border border-gray-300"
                >
                  <svg className="w-5 h-5" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                  </svg>
                  <span>Continue with Google</span>
                </button>

                {/* Toggle between login/signup */}
                <div className="text-center mt-3">
                  <button
                    type="button"
                    onClick={switchMode}
                    className="text-green-400 hover:text-green-300 text-sm font-medium"
                  >
                    {isSignUp 
                      ? 'Already have an account? Sign in' 
                      : 'Need an account? Sign up'
                    }
                  </button>
                </div>

                <p className="text-xs text-gray-500 text-center mt-3">
                  By continuing, you agree to our Terms of Service and Privacy Policy.
                </p>
              </form>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ProfileTab;