'use client';
import { X, ZoomIn, ZoomOut, RotateCw, Download } from 'lucide-react';
import React, { useState, useEffect } from 'react';

interface ImageEnlargementModalProps {
  isOpen: boolean;
  onClose: () => void;
  imageSrc: string;
  alt?: string;
  title?: string;
}

const ImageEnlargementModal: React.FC<ImageEnlargementModalProps> = ({
  isOpen,
  onClose,
  imageSrc,
  alt = 'Enlarged image',
  title = 'Image Preview'
}) => {
  const [scale, setScale] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Debug logging
  console.log('ImageEnlargementModal props:', { isOpen, imageSrc: imageSrc ? 'Present' : 'Missing', title });

  // Reset state when modal opens/closes
  useEffect(() => {
    if (isOpen) {
      setScale(1);
      setRotation(0);
      setPosition({ x: 0, y: 0 });
    }
  }, [isOpen]);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;
      
      switch (e.key) {
        case 'Escape':
          onClose();
          break;
        case '+':
        case '=':
          e.preventDefault();
          setScale(prev => Math.min(prev * 1.2, 5));
          break;
        case '-':
          e.preventDefault();
          setScale(prev => Math.max(prev / 1.2, 0.1));
          break;
        case 'r':
        case 'R':
          e.preventDefault();
          setRotation(prev => (prev + 90) % 360);
          break;
        case '0':
          e.preventDefault();
          setScale(1);
          setRotation(0);
          setPosition({ x: 0, y: 0 });
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({
      x: e.clientX - position.x,
      y: e.clientY - position.y
    });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    
    setPosition({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setScale(prev => Math.max(0.1, Math.min(5, prev * delta)));
  };

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = imageSrc;
    link.download = `image-${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/90 backdrop-blur-sm flex items-center justify-center z-[9999] p-4">
      <div className="relative w-full h-full max-w-7xl max-h-[95vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 bg-gray-900/80 backdrop-blur-sm border-b border-gray-700">
          <div className="flex items-center space-x-3">
            <h3 className="text-lg font-semibold text-white">{title}</h3>
            <div className="flex items-center space-x-2 text-sm text-gray-400">
              <span>Scale: {Math.round(scale * 100)}%</span>
              {rotation !== 0 && <span>• Rotation: {rotation}°</span>}
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            {/* Controls */}
            <div className="flex items-center space-x-1 bg-gray-800 rounded-lg p-1">
              <button
                onClick={() => setScale(prev => Math.max(0.1, prev / 1.2))}
                className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
                title="Zoom Out (-)"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <button
                onClick={() => setScale(1)}
                className="px-3 py-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors text-sm"
                title="Reset (0)"
              >
                100%
              </button>
              <button
                onClick={() => setScale(prev => Math.min(5, prev * 1.2))}
                className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
                title="Zoom In (+)"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
            </div>
            
            <button
              onClick={() => setRotation(prev => (prev + 90) % 360)}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
              title="Rotate (R)"
            >
              <RotateCw className="w-4 h-4" />
            </button>
            
            <button
              onClick={handleDownload}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
              title="Download"
            >
              <Download className="w-4 h-4" />
            </button>
            
            <button
              onClick={onClose}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
              title="Close (Esc)"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Image Container */}
        <div 
          className="flex-1 flex items-center justify-center overflow-hidden bg-gray-900"
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
        >
          <div
            className="relative cursor-grab active:cursor-grabbing select-none"
            style={{
              transform: `scale(${scale}) rotate(${rotation}deg) translate(${position.x / scale}px, ${position.y / scale}px)`,
              transformOrigin: 'center center',
              transition: isDragging ? 'none' : 'transform 0.2s ease-out'
            }}
            onMouseDown={handleMouseDown}
          >
            <img
              src={imageSrc}
              alt={alt}
              className="max-w-none"
              style={{
                maxWidth: 'none',
                maxHeight: 'none',
                userSelect: 'none',
                pointerEvents: 'none'
              }}
              draggable={false}
            />
          </div>
        </div>

        {/* Instructions */}
        <div className="p-4 bg-gray-900/80 backdrop-blur-sm border-t border-gray-700">
          <div className="flex items-center justify-center space-x-6 text-sm text-gray-400">
            <div className="flex items-center space-x-1">
              <kbd className="px-2 py-1 bg-gray-800 rounded text-xs">+</kbd>
              <span>Zoom In</span>
            </div>
            <div className="flex items-center space-x-1">
              <kbd className="px-2 py-1 bg-gray-800 rounded text-xs">-</kbd>
              <span>Zoom Out</span>
            </div>
            <div className="flex items-center space-x-1">
              <kbd className="px-2 py-1 bg-gray-800 rounded text-xs">R</kbd>
              <span>Rotate</span>
            </div>
            <div className="flex items-center space-x-1">
              <kbd className="px-2 py-1 bg-gray-800 rounded text-xs">0</kbd>
              <span>Reset</span>
            </div>
            <div className="flex items-center space-x-1">
              <kbd className="px-2 py-1 bg-gray-800 rounded text-xs">Esc</kbd>
              <span>Close</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageEnlargementModal;
