'use client';
import { CheckCircle } from 'lucide-react';
import React from 'react';

const HomeTab = () => {
  return (
    <div className="p-6 space-y-8">
    {/* Welcome Section */}
      <div className="text-center">
        <h2 className="text-2xl font-extrabold mb-2 uppercase tracking-wide">
          <span className="bg-gradient-to-r from-green-400 to-emerald-500 bg-clip-text text-transparent">
            SHROOM
          </span>
          <span className="bg-gradient-to-r from-yellow-300 to-amber-400 bg-clip-text text-transparent">
            IFY
          </span>
        </h2>
        <p className="text-gray-400">An Image-based Contamination Detection for Oyster Mushroom Fruiting Bags</p>
      </div>

      {/* Join Us Content */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        <div className="aspect-video bg-gradient-to-br from-green-600/20 to-blue-600/20 flex items-center justify-center">
          <img src="/banner_image.png" alt="Contamination Prevention" className="w-full h-full object-cover" />
        </div>
        <div className="p-4">
          <h3 className="text-lg font-semibold text-white mb-2 text-center">Identify Contamination Early and Prevent Bigger Problems Later!</h3>
          <p className="text-gray-400 text-sm mb-3 text-justify">
            Shroomify is your go-to application for mushroom cultivation-designed to detect contamination early.
            Using machine learning, Shroomify helps you monitor your fruiting bags, minimizing losses and maximizing productivity.
          </p>
          <div className="flex items-center text-green-400 text-sm">
            <CheckCircle className="w-4 h-4 mr-2" />
            <span>Designed for beginners and pros alike</span>
          </div>

            {/* Join KabuTeam Button */}
          <div className="text-center">
            <button className="bg-gradient-to-r from-green-600 to-green-500 hover:from-green-700 hover:to-green-600 text-white font-semibold py-3 px-8 rounded-lg shadow-lg transform hover:scale-105 transition-all duration-200 flex items-center space-x-2 mx-auto mt-5">
              <span>🍄</span>
              <span>Join the KabuTeam</span>
            </button>
          </div>

        </div>
      </div>

      {/* Cultivation Methods */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-4">Popular Cultivation Methods</h3>
        <div className="space-y-3">
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-start space-x-3">
              <div className="w-12 h-12 bg-gray-700 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-lg">🌾</span>
              </div>
              <div className="flex-1">
                <h4 className="text-white font-medium mb-1">Straw Substrate Method</h4>
                <p className="text-gray-400 text-sm mb-2">
                  Cost-effective method using pasteurized straw. Ideal for oyster mushrooms and wine cap mushrooms.
                </p>
                <div className="flex items-center text-xs">
                  <span className="bg-green-600/20 text-green-400 px-2 py-1 rounded mr-2">Beginner Friendly</span>
                  <span className="text-gray-500">Success Rate: 80%</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-start space-x-3">
              <div className="w-12 h-12 bg-gray-700 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-lg">🪵</span>
              </div>
              <div className="flex-1">
                <h4 className="text-white font-medium mb-1">Hardwood Sawdust</h4>
                <p className="text-gray-400 text-sm mb-2">
                  Premium method for shiitake, lion&apos;s mane, and reishi. Requires supplemented sawdust and sterilization.
                </p>
                <div className="flex items-center text-xs">
                  <span className="bg-yellow-600/20 text-yellow-400 px-2 py-1 rounded mr-2">Intermediate</span>
                  <span className="text-gray-500">Success Rate: 75%</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-start space-x-3">
              <div className="w-12 h-12 bg-gray-700 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-lg">🌱</span>
              </div>
              <div className="flex-1">
                <h4 className="text-white font-medium mb-1">Agar Culture Work</h4>
                <p className="text-gray-400 text-sm mb-2">
                  Advanced technique for isolating pure strains and maintaining genetic consistency in cultivation.
                </p>
                <div className="flex items-center text-xs">
                  <span className="bg-red-600/20 text-red-400 px-2 py-1 rounded mr-2">Advanced</span>
                  <span className="text-gray-500">Success Rate: 90%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Growth Stages */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-4">Understanding Growth Stages</h3>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-3 mb-2">
              <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white text-sm font-bold">1</div>
              <h4 className="text-white font-medium">Inoculation (Days 1-7)</h4>
            </div>
            <p className="text-gray-400 text-sm ml-11">
              Spores or liquid culture are introduced to sterile substrate. Critical contamination prevention period.
            </p>
          </div>

          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-3 mb-2">
              <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center text-white text-sm font-bold">2</div>
              <h4 className="text-white font-medium">Colonization (Days 7-21)</h4>
            </div>
            <p className="text-gray-400 text-sm ml-11">
              Mycelium spreads through substrate. Maintain optimal temperature and humidity without light exposure.
            </p>
          </div>

          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-3 mb-2">
              <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white text-sm font-bold">3</div>
              <h4 className="text-white font-medium">Fruiting (Days 21-35)</h4>
            </div>
            <p className="text-gray-400 text-sm ml-11">
              Introduce fresh air exchange, light cycles, and maintain high humidity for mushroom formation.
            </p>
          </div>
        </div>
      </div>

      {/* Essential Parameters */}
      <div className="bg-gradient-to-r from-blue-600/10 to-purple-600/10 rounded-lg p-4 border border-blue-600/20">
        <h3 className="text-lg font-semibold text-white mb-3">🌡️ Optimal Growing Conditions</h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-blue-400 font-medium">Temperature:</span>
            <p className="text-gray-300">65-75°F (18-24°C)</p>
          </div>
          <div>
            <span className="text-blue-400 font-medium">Humidity:</span>
            <p className="text-gray-300">80-95% RH</p>
          </div>
          <div>
            <span className="text-blue-400 font-medium">Air Exchange:</span>
            <p className="text-gray-300">4-6 times per hour</p>
          </div>
          <div>
            <span className="text-blue-400 font-medium">Light Cycle:</span>
            <p className="text-gray-300">12 hours indirect light</p>
          </div>
        </div>
      </div>

      {/* Knowledge Tip */}
      <div className="bg-gradient-to-r from-green-600/10 to-blue-600/10 rounded-lg p-4 border border-green-600/20">
        <h3 className="text-lg font-semibold text-white mb-2">📚 Did You Know?</h3>
        <p className="text-gray-300 text-sm">
          Mushrooms are more closely related to animals than plants! They obtain nutrients by breaking down organic matter, 
          just like animals do, rather than producing their own food through photosynthesis like plants.
        </p>
      </div>
    </div>
  );
};

export default HomeTab;
