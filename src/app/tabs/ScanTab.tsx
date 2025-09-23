'use client';
import { Camera, Video, Zap, Upload, CheckCircle, AlertTriangle, XCircle, X, Eye } from 'lucide-react';
import React, { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import ErrorModal from "@/app/tabs/ErrorModal";
import { useAuth } from "@/lib/AuthContext";
import { supabase } from "@/lib/supabaseClient";
import ImageEnlargementModal from './ImageEnlargementModal';

// Interface for scan data stored in localStorage
interface ScanData {
  id: string;
  date_logged: string;
  detected_disease: number;
  email: string;
  confidence: number;
  image: string;
  synced: boolean;
}

// ResultPopup Component (inline)
const ResultPopup = ({ result, previewImage, confidence, isOpen, onClose, isLoggedIn, userEmail, onSaveResult, isSaving, saveMessage, isSaved, setIsSaved }: { 
  result: number | string | null, 
  previewImage: string | null, 
  confidence: number | null, 
  isOpen: boolean, 
  onClose: () => void, 
  isLoggedIn: boolean,
  userEmail: string | null,
  onSaveResult: (result: number, image: string, confidence: number, email: string) => Promise<void>,
  isSaving: boolean,
  saveMessage: string | null,
  isSaved: boolean,
  setIsSaved: (value: boolean) => void
}) => {
  const [enlargedImage, setEnlargedImage] = useState<{ src: string; title: string } | null>(null);
  
  console.log('ResultPopup - isLoggedIn:', isLoggedIn); // Debug log
  if (!isOpen) return null;

  // Condition statements for different results
  const getResultData = (resultCode: number | string | null) => {
    // Handle special case for no fruiting bag detected
    if (resultCode === 'no_fruiting_bag') {
      return {
        status: 'No Fruiting Bag Detected',
        icon: XCircle,
        iconColor: 'text-orange-500',
        bgColor: 'bg-orange-600/10',
        borderColor: 'border-orange-600/30',
        message: 'No fruiting bag detected in the image',
        description: 'The system could not detect a fruiting bag in the provided image. Please ensure you are scanning a clear image of a fruiting bag.',
        recommendations: [
          'Ensure the image shows a clear view of a fruiting bag',
          'Check that the bag is well-lit and in focus',
          'Try taking the photo from a different angle',
          'Make sure the entire bag is visible in the frame'
        ],
        severity: 'info',
        showSaveButton: false // Don't show save button for this case
      };
    }
    
    switch (resultCode) {
      case 0:
        return {
          status: 'Healthy',
          icon: CheckCircle,
          iconColor: 'text-green-500',
          bgColor: 'bg-green-600/10',
          borderColor: 'border-green-600/30',
          message: 'No contamination detected',
          description: 'Your fruiting bag appears to be healthy and free from contamination.',
          recommendations: [
            'Continue with regular monitoring',
            'Maintain proper humidity and temperature',
            'Keep sterile conditions'
          ],
          severity: 'success'
        };
      case 1:
        return {
          status: 'Green Mold Detected',
          icon: AlertTriangle,
          iconColor: 'text-yellow-500',
          bgColor: 'bg-yellow-600/10',
          borderColor: 'border-yellow-600/30',
          message: 'Green mold contamination found',
          description: 'Green mold (likely Trichoderma) has been detected in your fruiting bag.',
          recommendations: [
            'Isolate the bag immediately',
            'Do not open indoors to prevent spore spread',
            'Consider disposing of the contaminated bag',
            'Review sterile techniques'
          ],
          severity: 'warning'
        };
      case 2:
        return {
          status: 'Black Mold Detected',
          icon: XCircle,
          iconColor: 'text-red-500',
          bgColor: 'bg-red-600/10',
          borderColor: 'border-red-600/30',
          message: 'Black mold contamination found',
          description: 'Black mold contamination has been detected. This requires immediate attention.',
          recommendations: [
            'Quarantine the bag immediately',
            'Dispose of the bag safely outdoors',
            'Disinfect the growing area thoroughly',
            'Review and improve sterile procedures',
            'Consider professional consultation'
          ],
          severity: 'danger'
        };
      default:
        return {
          status: 'Unknown Result',
          icon: AlertTriangle,
          iconColor: 'text-gray-500',
          bgColor: 'bg-gray-600/10',
          borderColor: 'border-gray-600/30',
          message: 'Unable to determine result',
          description: 'The analysis could not be completed properly.',
          recommendations: [
            'Try scanning again with better lighting',
            'Ensure the image is clear and focused',
            'Contact support if issues persist'
          ],
          severity: 'info'
        };
    }
  };

  const resultData = getResultData(result);
  const IconComponent = resultData.icon;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-2000 p-4">
      <div className="bg-gray-900 rounded-2xl border border-gray-700 max-w-md w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div className="flex items-center space-x-3">
            <div className={`w-10 h-10 rounded-full ${resultData.bgColor} ${resultData.borderColor} border flex items-center justify-center`}>
              <IconComponent className={`w-6 h-6 ${resultData.iconColor}`} />
            </div>
            <div className="flex-1">
              <h3 className="text-xl font-bold text-white">{resultData.status}</h3>
              {confidence !== null && (
                <div className="mt-1 flex items-center space-x-2">
                  <span className="text-xs text-gray-500">Confidence:</span>
                  <div className="flex items-center space-x-1">
                    <div className={`w-2 h-2 rounded-full ${
                      confidence >= 0.8 ? 'bg-green-400' :
                      confidence >= 0.6 ? 'bg-yellow-400' : 'bg-red-400'
                    }`}></div>
                    <span className={`text-xs font-medium ${
                      confidence >= 0.8 ? 'text-green-400' :
                      confidence >= 0.6 ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {Math.round(confidence * 100)}%
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-full bg-gray-800 hover:bg-gray-700 flex items-center justify-center transition-colors"
          >
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Preview Image */}
          {previewImage && (
            <div>
              <div 
                className="relative bg-gray-800 rounded-lg border border-gray-700 overflow-hidden cursor-pointer hover:bg-gray-700 transition-colors group"
                onClick={() => {
                  console.log('Preview image clicked');
                  setEnlargedImage({
                    src: `data:image/jpeg;base64,${previewImage}`,
                    title: `Scan Result - ${resultData.status}`
                  });
                }}
              >
                <img
                  src={`data:image/jpeg;base64,${previewImage}`}
                  alt="Scanned fruiting bag"
                  className="w-full h-48 object-cover pointer-events-none"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent pointer-events-none"></div>
                {/* Hover overlay */}
                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
                  <Eye className="w-8 h-8 text-white" />
                </div>
              </div>
            </div>
          )}

          {/* Description */}
          <div className={`${resultData.bgColor} ${resultData.borderColor} border rounded-lg p-4`}>
            <p className="text-gray-300 text-sm leading-relaxed">
              {resultData.description}
            </p>
          </div>

          {/* Recommendations */}
          <div>
            <h4 className="text-lg font-semibold text-white mb-3">Recommendations</h4>
            <div className="space-y-2">
              {resultData.recommendations.map((recommendation, index) => (
                <div key={index} className="flex items-start space-x-3">
                  <div className={`w-2 h-2 rounded-full mt-2 flex-shrink-0 ${
                    resultData.severity === 'success' ? 'bg-green-400' :
                    resultData.severity === 'warning' ? 'bg-yellow-400' :
                    resultData.severity === 'danger' ? 'bg-red-400' : 'bg-gray-400'
                  }`}></div>
                  <p className="text-gray-300 text-sm">{recommendation}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Save Status Message */}
          {saveMessage && (
            <div className={`p-3 rounded-lg border ${
              saveMessage.includes('successfully') 
                ? 'bg-green-600/10 border-green-600/30 text-green-400'
                : 'bg-red-600/10 border-red-600/30 text-red-400'
            }`}>
              <p className="text-sm font-medium">{saveMessage}</p>
            </div>
          )}

          {/* Action Buttons */}
          <div className={`flex space-x-3 pt-4 ${(isLoggedIn && resultData.showSaveButton !== false) ? '' : 'justify-center'}`}>
            <button
              onClick={onClose}
              className={`${(isLoggedIn && resultData.showSaveButton !== false) ? 'flex-1' : 'w-full'} bg-gray-800 hover:bg-gray-700 text-white py-3 px-4 rounded-lg transition-colors font-medium`}
            >
              Close
            </button>
            {isLoggedIn === true && !isSaved && resultData.showSaveButton !== false && (
              <button
                onClick={async () => {
                  if (result !== null && previewImage && confidence !== null && userEmail && typeof result === 'number') {
                    try {
                      await onSaveResult(result, previewImage, confidence, userEmail);
                      console.log('Result saved successfully');
                      setIsSaved(true);
                      // Don't close immediately, let user see the success message
                    } catch (error) {
                      console.error('Failed to save result:', error);
                    }
                  }
                }}
                disabled={isSaving}
                className={`flex-1 py-3 px-4 rounded-lg transition-colors font-medium flex items-center justify-center space-x-2 ${
                  isSaving 
                    ? 'bg-gray-600 cursor-not-allowed text-gray-300'
                    : resultData.severity === 'success' 
                    ? 'bg-green-600 hover:bg-green-500 text-white'
                    : resultData.severity === 'warning'
                    ? 'bg-yellow-600 hover:bg-yellow-500 text-white'
                    : resultData.severity === 'danger'
                    ? 'bg-red-600 hover:bg-red-500 text-white'
                    : 'bg-blue-600 hover:bg-blue-500 text-white'
                }`}
              >
                {isSaving ? (
                  <>
                    <div className="w-4 h-4 border-2 border-gray-300 border-t-transparent rounded-full animate-spin"></div>
                    <span>Saving...</span>
                  </>
                ) : (
                  'Save Result'
                )}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Image Enlargement Modal */}
      <ImageEnlargementModal
        isOpen={enlargedImage !== null}
        onClose={() => setEnlargedImage(null)}
        imageSrc={enlargedImage?.src || ''}
        title={enlargedImage?.title || 'Image Preview'}
        alt="Enlarged scan image"
      />
    </div>
  );
};

// Main ScanTab Component
const ScanTab = () => {
  const { isLoggedIn, user } = useAuth();
  console.log('ScanTab - isLoggedIn:', isLoggedIn); // Debug log
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const webcamRef = useRef<Webcam>(null);
  
  // New state for result popup
  const [showResult, setShowResult] = useState(false);
  const [scanResult, setScanResult] = useState<number | string | null>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [showError, setShowError] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [isSyncing, setIsSyncing] = useState(false);
  const [isSaved, setIsSaved] = useState(false);

  // Auto-sync localStorage data to database on component mount
  useEffect(() => {
    const autoSync = async () => {
      try {
        const scans = JSON.parse(localStorage.getItem('shroomify_scans') || '[]');
        const unsyncedScans = scans.filter((scan: ScanData) => !scan.synced);
        
        if (unsyncedScans.length > 0) {
          console.log(`Found ${unsyncedScans.length} unsynced scans, attempting to sync...`);
          await syncToDatabase();
        }
      } catch (error) {
        console.error('Auto-sync failed:', error);
      }
    };

    autoSync();
  }, []);

  // Function to save scan result to localStorage
  const saveToLocalStorage = (result: number, image: string, confidence: number, email: string) => {
    try {
      const scanData = {
        id: Date.now().toString(),
        date_logged: new Date().toISOString(),
        detected_disease: result,
        email: email,
        confidence: confidence,
        image: image, // Store as base64 string in localStorage
        synced: false // Flag to track if synced to database
      };

      // Get existing scans from localStorage
      const existingScans = JSON.parse(localStorage.getItem('shroomify_scans') || '[]');
      
      // Add new scan
      existingScans.push(scanData);
      
      // Keep only last 50 scans to prevent localStorage from getting too large
      if (existingScans.length > 50) {
        existingScans.splice(0, existingScans.length - 50);
      }
      
      // Save back to localStorage
      localStorage.setItem('shroomify_scans', JSON.stringify(existingScans));
      
      console.log('Saved scan result to localStorage:', scanData);
      return scanData;
    } catch (error) {
      console.error('Failed to save to localStorage:', error);
      throw error;
    }
  };

  // Function to sync localStorage data to database
  const syncToDatabase = async () => {
    setIsSyncing(true);
    try {
      const scans = JSON.parse(localStorage.getItem('shroomify_scans') || '[]');
      const unsyncedScans = scans.filter((scan: ScanData) => !scan.synced);
      
      if (unsyncedScans.length === 0) {
        console.log('No unsynced scans to upload');
        setSaveMessage('All scans are already synced!');
        setTimeout(() => setSaveMessage(null), 3000);
        return;
      }

      console.log(`Syncing ${unsyncedScans.length} scans to database...`);
      setSaveMessage(`Syncing ${unsyncedScans.length} scans to cloud...`);

      let syncedCount = 0;
      for (const scan of unsyncedScans) {
        try {
          // Convert base64 image to binary data for bytea storage
          const binaryString = atob(scan.image);
          const bytes = new Uint8Array(binaryString.length);
          for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
          }

          const { error } = await supabase
            .from('Logs')
            .insert({
              date_logged: scan.date_logged,
              image: bytes,
              detected_disease: scan.detected_disease,
              email: scan.email,
              confidence: scan.confidence
            });

          if (error) {
            console.error('Error syncing scan to database:', error);
            // Don't mark as synced if there was an error
            continue;
          }

          // Mark as synced
          scan.synced = true;
          syncedCount++;
          console.log('Successfully synced scan to database:', scan.id);
        } catch (error) {
          console.error('Failed to sync individual scan:', error);
        }
      }

      // Update localStorage with synced status
      localStorage.setItem('shroomify_scans', JSON.stringify(scans));
      console.log('Sync to database completed');
      
      if (syncedCount > 0) {
        setSaveMessage(`Successfully synced ${syncedCount} scans to cloud!`);
      } else {
        setSaveMessage('Sync failed - scans remain saved locally');
      }
      
      setTimeout(() => setSaveMessage(null), 5000);
    } catch (error) {
      console.error('Failed to sync to database:', error);
      setSaveMessage('Sync failed - scans remain saved locally');
      setTimeout(() => setSaveMessage(null), 5000);
    } finally {
      setIsSyncing(false);
    }
  };


  // Function to save scan result (localStorage first, then database)
  const saveScanResult = async (result: number, image: string, confidence: number, email: string) => {
    setIsSaving(true);
    setSaveMessage(null);
    
    try {
      console.log('Saving scan result:', { result, confidence, email, imageLength: image.length });
      
      // Step 1: Save to localStorage first (always succeeds)
      const savedData = saveToLocalStorage(result, image, confidence, email);
      console.log('Successfully saved to localStorage');
      setSaveMessage('Saving . . .');
      
      // Step 2: Try to sync to database (may fail, but that's okay)
      try {
        // Convert base64 image to binary data for bytea storage
        const binaryString = atob(image);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        
        console.log('Attempting to sync to database...');

        const { data, error } = await supabase
          .from('Logs')
          .insert({
            date_logged: savedData.date_logged,
            image: bytes,
            detected_disease: result,
            email: email,
            confidence: confidence
          });

        if (error) {
          console.error('Database sync failed:', error);
          console.error('Error details:', {
            message: error.message,
            details: error.details,
            hint: error.hint,
            code: error.code
          });
          setSaveMessage('Saving . . .');
        } else {
          console.log('Successfully synced to database:', data);
          // Mark as synced in localStorage
          const scans = JSON.parse(localStorage.getItem('shroomify_scans') || '[]');
          const updatedScans = scans.map((scan: ScanData) => 
            scan.id === savedData.id ? { ...scan, synced: true } : scan
          );
          localStorage.setItem('shroomify_scans', JSON.stringify(updatedScans));
          setSaveMessage('Result saved successfully!');
          // Close popup after showing success message
          setTimeout(() => {
            setShowResult(false);
          }, 2000);
        }
      } catch (dbError) {
        console.error('Database sync error:', dbError);
        setSaveMessage('Result saved locally! (Database sync failed - will retry later)');
        // Close popup after showing success message
        setTimeout(() => {
          setShowResult(false);
        }, 2000);
      }
      
      // Clear success message after 5 seconds
      setTimeout(() => {
        setSaveMessage(null);
      }, 5000);
      
    } catch (error) {
      console.error('Failed to save scan result:', error);
      setSaveMessage('Failed to save result. Please try again.');
      throw error;
    } finally {
      setIsSaving(false);
    }
  };

  const handleSnap = async () => {
  if (webcamRef.current) {
    const imageSrc = webcamRef.current.getScreenshot();

    if (!imageSrc) {
      setShowError(true); 
      return;
    }

    console.log("Captured image:", imageSrc);
    setIsScanning(true);

    try {
      // Convert base64 to blob
      const res = await fetch(imageSrc);
      const blob = await res.blob();

      // Prepare form data
      const formData = new FormData();
      formData.append('image', blob, 'snapshot.jpg');

      // Send to backend
      const response = await fetch('https://reliably-one-kiwi.ngrok-free.app/api/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      console.log(result);

      // Check for error response (no fruiting bag detected)
      if (result.error && result.status === 'error') {
        setScanResult('no_fruiting_bag'); // Special case for no fruiting bag
        setPreviewImage(result.image || null);
        setConfidence(null);
        setIsSaved(false);
        setShowResult(true);
        return;
      }

      // Store and show result
      setScanResult(result.prediction ?? result.result);
      setPreviewImage(result.image);
      setConfidence(result.confidence);
      setIsSaved(false);
      setShowResult(true);

    } catch (error) {
      console.error('Scan failed:', error);
      setShowError(true); 
    } finally {
      setIsScanning(false);
    }
  }
};

  const handleUpload = () => {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.onchange = async (e) => {
      const target = e.target as HTMLInputElement;
      if (target && target.files && target.files[0]) {
        const file = target.files[0];
        setIsScanning(true);

        const formData = new FormData();
        formData.append('image', file);

        try {
          const response = await fetch('https://reliably-one-kiwi.ngrok-free.app/api/upload', {
            method: 'POST',
            body: formData,
          });

          const result = await response.json();
          console.log(result);
          
          // Check for error response (no fruiting bag detected)
          if (result.error && result.status === 'error') {
            setScanResult('no_fruiting_bag'); // Special case for no fruiting bag
            setPreviewImage(result.image || null);
            setConfidence(null);
            setIsSaved(false);
            setShowResult(true);
            return;
          }
          
          setScanResult(result.prediction || result.result);
          setPreviewImage(result.image);
          setConfidence(result.confidence);
          setIsSaved(false);
          setShowResult(true);
          
        } catch (err) {
          console.error('Upload failed:', err);
          setShowError(true);
        } finally {
          setIsScanning(false);
        }
      }
    };
    fileInput.click();
  };

  return (
    <div className="p-6 space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-extrabold mb-2 uppercase tracking-wide">
          <span className="bg-gradient-to-r from-green-400 to-emerald-500 bg-clip-text text-transparent">
            SCAN
          </span>
          <span className="bg-gradient-to-r from-yellow-300 to-amber-400 bg-clip-text text-transparent">
            NOW
          </span>
        </h2>
        <p className="text-gray-400">Detect contamination in your fruiting bags</p>
      </div>

      {/* Toggle */}
      <div className="flex justify-center mb-6">
        <div className="bg-gray-800 p-1 rounded-full border border-gray-700">
          <div className="flex items-center space-x-1">
            <button
              onClick={() => setIsLiveMode(false)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-full transition-all duration-300 ${
                !isLiveMode
                  ? 'bg-gradient-to-r from-green-600 to-green-500 text-white shadow-lg'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Camera className="w-4 h-4" />
              <span className="text-sm font-medium">Static</span>
            </button>
            <button
              onClick={() => setIsLiveMode(true)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-full transition-all duration-300 ${
                isLiveMode
                  ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-lg'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Video className="w-4 h-4" />
              <span className="text-sm font-medium">Live Feed</span>
            </button>
          </div>
        </div>
      </div>

      {/* Camera Interface */}
      <div className="relative">
        <div className="bg-gray-800 rounded-2xl border-2 border-gray-700 overflow-hidden aspect-square max-w-lg mx-auto relative">
          {/* Viewfinder */}
          <div className="w-full h-full bg-gradient-to-br from-gray-900 to-gray-800 flex items-center justify-center relative">
            {isScanning ? (
              <div className="text-center">
                <div className="w-16 h-16 border-4 border-green-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-green-400 font-medium">Analyzing...</p>
              </div>
            ) : (
              <>
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  className="absolute top-0 left-0 w-full h-full object-cover"
                  videoConstraints={{
                    facingMode: 'environment',
                  }}
                />
                {/* Overlay UI */}
                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                  <p className="text-gray-400 font-medium bg-black/50 px-2 rounded">
                    {isLiveMode ? '' : ''}
                  </p>
                </div>
              </>
            )}

            {/* Corner Guides */}
            <div className="absolute top-4 left-4 w-6 h-6 border-l-2 border-t-2 border-green-400 rounded-tl-lg"></div>
            <div className="absolute top-4 right-4 w-6 h-6 border-r-2 border-t-2 border-green-400 rounded-tr-lg"></div>
            <div className="absolute bottom-4 left-4 w-6 h-6 border-l-2 border-b-2 border-green-400 rounded-bl-lg"></div>
            <div className="absolute bottom-4 right-4 w-6 h-6 border-r-2 border-b-2 border-green-400 rounded-br-lg"></div>

            {/* Live Mode Indicator */}
            {isLiveMode && !isScanning && (
              <div className="absolute top-4 left-1/2 transform -translate-x-1/2">
                <div className="flex items-center space-x-2 bg-red-600/20 border border-red-600/30 rounded-full px-3 py-1">
                  <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                  <span className="text-red-400 text-xs font-medium">LIVE</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Buttons */}
        <div className="flex justify-center items-center mt-8 space-x-6">
          {!isLiveMode && (
            <button
              onClick={handleUpload}
              disabled={isScanning}
              className={`w-16 h-16 rounded-full border-4 flex items-center justify-center transition-all duration-200 ${
                isScanning
                  ? 'border-gray-600 bg-gray-700 cursor-not-allowed'
                  : 'border-amber-400 bg-amber-600 hover:bg-amber-500 hover:border-amber-300 transform hover:scale-105 shadow-lg shadow-amber-600/25'
              }`}
            >
              {isScanning ? (
                <div className="w-5 h-5 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
              ) : (
                <Upload className="w-6 h-6 text-white" />
              )}
            </button>
          )}
          <button
            onClick={handleSnap}
            disabled={isScanning}
            className={`w-20 h-20 rounded-full border-4 flex items-center justify-center transition-all duration-200 ${
              isScanning
                ? 'border-gray-600 bg-gray-700 cursor-not-allowed'
                : isLiveMode
                ? 'border-blue-400 bg-blue-600 hover:bg-blue-500 hover:border-blue-300 transform hover:scale-105 shadow-lg shadow-blue-600/25'
                : 'border-green-400 bg-green-600 hover:bg-green-500 hover:border-green-300 transform hover:scale-105 shadow-lg shadow-green-600/25'
            }`}
          >
            {isScanning ? (
              <div className="w-6 h-6 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <Camera className="w-8 h-8 text-white" />
            )}
          </button>
        </div>

        {/* Action Text */}
        <div className="text-center mt-4">
          <p className="text-gray-400 text-sm">
            {isScanning
              ? 'Processing image for contamination...'
              : isLiveMode
              ? 'Tap to capture from live feed'
              : 'Tap to take a photo or upload an image'}
          </p>
        </div>
      </div>

      {/* Scanning Tips */}
      <div className="bg-gradient-to-r from-green-600/10 to-blue-600/10 rounded-lg p-4 border border-green-600/20">
        <h3 className="text-lg font-semibold text-white mb-3 flex items-center">
          <Zap className="w-5 h-5 mr-2 text-yellow-400" />
          Scanning Tips
        </h3>
        <div className="space-y-2 text-sm">
          <div className="flex items-start space-x-2">
            <div className="w-1.5 h-1.5 bg-green-400 rounded-full mt-2 flex-shrink-0"></div>
            <p className="text-gray-300">Ensure good lighting on your fruiting bag</p>
          </div>
          <div className="flex items-start space-x-2">
            <div className="w-1.5 h-1.5 bg-green-400 rounded-full mt-2 flex-shrink-0"></div>
            <p className="text-gray-300">Hold camera steady and focus on the entire bag</p>
          </div>
          <div className="flex items-start space-x-2">
            <div className="w-1.5 h-1.5 bg-green-400 rounded-full mt-2 flex-shrink-0"></div>
            <p className="text-gray-300">Clean the bag surface for better detection accuracy</p>
          </div>
          <div className="flex items-start space-x-2">
            <div className="w-1.5 h-1.5 bg-green-400 rounded-full mt-2 flex-shrink-0"></div>
            <p className="text-gray-300">Use live feed mode for real-time positioning</p>
          </div>
          {!isLiveMode && (
            <div className="flex items-start space-x-2">
              <div className="w-1.5 h-1.5 bg-amber-400 rounded-full mt-2 flex-shrink-0"></div>
              <p className="text-gray-300">Upload existing photos from your gallery for analysis</p>
            </div>
          )}
        </div>
      </div>

      {/* Result Popup */}
      <ResultPopup 
        result={scanResult} 
        previewImage={previewImage}
        confidence={confidence}
        isOpen={showResult} 
        onClose={() => setShowResult(false)}
        isLoggedIn={isLoggedIn}
        userEmail={user?.email || null}
        onSaveResult={saveScanResult}
        isSaving={isSaving}
        saveMessage={saveMessage}
        isSaved={isSaved}
        setIsSaved={setIsSaved}
      />

       {/* Error Modal */}
      <ErrorModal
        isOpen={showError}
        onClose={() => setShowError(false)}
        onRetry={() => window.location.reload()}
        title="Unexpected Error!"
        message="Oh no! Cannot connect to the backend server. Please try again or contact the developers."
      />
    </div>
  );
};

export default ScanTab;