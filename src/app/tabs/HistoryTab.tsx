'use client';
import { Calendar, Clock, AlertTriangle, CheckCircle, Eye, Trash2, Filter, Search, Download, ChevronLeft, ChevronRight } from 'lucide-react';
import React, { useState, useMemo, useCallback } from 'react';

// Types for better type safety and scalability
interface HistoryLog {
  id: string;
  timestamp: string;
  bagId: string;
  status: 'healthy' | 'contaminated';
  confidence: number;
  contaminationType?: string;
  imageUrl?: string;
}

interface FilterOptions {
  status: 'all' | 'healthy' | 'contaminated';
  dateRange: 'all' | 'today' | 'week' | 'month';
  minConfidence: number;
  searchTerm: string;
}

interface PaginationConfig {
  currentPage: number;
  itemsPerPage: number;
  totalItems: number;
}

// Configuration constants for easy scaling
const CONFIG = {
  ITEMS_PER_PAGE_OPTIONS: [10, 25, 50, 100],
  DEFAULT_ITEMS_PER_PAGE: 10,
  CONFIDENCE_THRESHOLDS: {
    HIGH: 90,
    MEDIUM: 75,
    LOW: 60
  },
  EXPORT_FORMATS: ['csv', 'json', 'pdf'] as const
};

const HistoryTab = () => {
  // State management for scalability
  const [filters, setFilters] = useState<FilterOptions>({
    status: 'all',
    dateRange: 'all',
    minConfidence: 0,
    searchTerm: ''
  });

  const [pagination, setPagination] = useState<PaginationConfig>({
    currentPage: 1,
    itemsPerPage: CONFIG.DEFAULT_ITEMS_PER_PAGE,
    totalItems: 0
  });

  const [selectedLogs, setSelectedLogs] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState(false);
  const [sortConfig, setSortConfig] = useState<{
    field: keyof HistoryLog;
    direction: 'asc' | 'desc';
  }>({
    field: 'timestamp',
    direction: 'desc'
  });

  // Mock data - in production this would come from an API/database
  const mockHistoryLogs: HistoryLog[] = useMemo(() => 
    Array.from({ length: 150 }, (_, index) => ({
      id: `log-${index + 1}`,
      timestamp: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
      bagId: `BAG-${String(index + 1).padStart(3, '0')}`,
      status: Math.random() > 0.3 ? 'healthy' : 'contaminated',
      confidence: Math.floor(Math.random() * 40 + 60),
      contaminationType: Math.random() > 0.6 ? ['Trichoderma', 'Penicillium', 'Bacterial Wet Spot', 'Cobweb Mold'][Math.floor(Math.random() * 4)] : undefined,
      imageUrl: `/scan${(index % 10) + 1}.jpg`
    })), []
  );

  // Filtered and sorted data with performance optimization
  const processedData = useMemo(() => {
    let filtered = [...mockHistoryLogs];

    // Apply filters
    if (filters.status !== 'all') {
      filtered = filtered.filter(log => log.status === filters.status);
    }

    if (filters.searchTerm) {
      const searchLower = filters.searchTerm.toLowerCase();
      filtered = filtered.filter(log => 
        log.bagId.toLowerCase().includes(searchLower) ||
        log.contaminationType?.toLowerCase().includes(searchLower)
      );
    }

    if (filters.minConfidence > 0) {
      filtered = filtered.filter(log => log.confidence >= filters.minConfidence);
    }

    if (filters.dateRange !== 'all') {
      const now = new Date();
      const cutoffDate = new Date();
      
      switch (filters.dateRange) {
        case 'today':
          cutoffDate.setHours(0, 0, 0, 0);
          break;
        case 'week':
          cutoffDate.setDate(now.getDate() - 7);
          break;
        case 'month':
          cutoffDate.setMonth(now.getMonth() - 1);
          break;
      }
      
      filtered = filtered.filter(log => new Date(log.timestamp) >= cutoffDate);
    }

    // Apply sorting
    filtered.sort((a, b) => {
      const aValue = a[sortConfig.field];
      const bValue = b[sortConfig.field];
      
      // Handle undefined values
      if (aValue == null && bValue == null) return 0;
      if (aValue == null) return sortConfig.direction === 'asc' ? -1 : 1;
      if (bValue == null) return sortConfig.direction === 'asc' ? 1 : -1;
      
      if (aValue < bValue) return sortConfig.direction === 'asc' ? -1 : 1;
      if (aValue > bValue) return sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    });

    return filtered;
  }, [mockHistoryLogs, filters, sortConfig]);

  // Paginated data
  const paginatedData = useMemo(() => {
    const startIndex = (pagination.currentPage - 1) * pagination.itemsPerPage;
    const endIndex = startIndex + pagination.itemsPerPage;
    return processedData.slice(startIndex, endIndex);
  }, [processedData, pagination.currentPage, pagination.itemsPerPage]);

  // Update total items when processed data changes
  React.useEffect(() => {
    setPagination(prev => ({ ...prev, totalItems: processedData.length, currentPage: 1 }));
  }, [processedData.length]);

  // Statistics calculation
  const statistics = useMemo(() => {
    const total = processedData.length;
    const healthy = processedData.filter(log => log.status === 'healthy').length;
    const contaminated = total - healthy;
    const avgConfidence = total > 0 
      ? Math.round(processedData.reduce((sum, log) => sum + log.confidence, 0) / total)
      : 0;
    
    return { total, healthy, contaminated, avgConfidence };
  }, [processedData]);

  // Event handlers
  const handleFilterChange = useCallback(
    <K extends keyof FilterOptions>(key: K, value: FilterOptions[K]) => {
      setFilters(prev => ({ ...prev, [key]: value }));
    },
    []
  );

  const handleSort = useCallback((field: keyof HistoryLog) => {
    setSortConfig(prev => ({
      field,
      direction: prev.field === field && prev.direction === 'asc' ? 'desc' : 'asc'
    }));
  }, []);

  const handlePageChange = useCallback((page: number) => {
    setPagination(prev => ({ ...prev, currentPage: page }));
  }, []);

  const handleItemsPerPageChange = useCallback((itemsPerPage: number) => {
    setPagination(prev => ({ ...prev, itemsPerPage, currentPage: 1 }));
  }, []);

  const handleBulkAction = useCallback((action: 'delete' | 'export', logIds: string[]) => {
    setIsLoading(true);
    // Simulate API call
    setTimeout(() => {
      console.log(`Performing ${action} on logs:`, logIds);
      setSelectedLogs(new Set());
      setIsLoading(false);
    }, 1000);
  }, []);

  const handleExport = useCallback((format: typeof CONFIG.EXPORT_FORMATS[number]) => {
    const dataToExport = selectedLogs.size > 0 
      ? processedData.filter(log => selectedLogs.has(log.id))
      : processedData;
    
    console.log(`Exporting ${dataToExport.length} records as ${format}`);
    // Implementation would handle actual export logic
  }, [processedData, selectedLogs]);

  // Utility functions
  const getStatusIcon = (status: HistoryLog['status']) => {
    return status === 'healthy' ? (
      <CheckCircle className="w-5 h-5 text-green-400" />
    ) : (
      <AlertTriangle className="w-5 h-5 text-red-400" />
    );
  };

  const getStatusColor = (status: HistoryLog['status']) => {
    return status === 'healthy' 
      ? 'bg-green-600/20 text-green-400 border-green-600/30' 
      : 'bg-red-600/20 text-red-400 border-red-600/30';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= CONFIG.CONFIDENCE_THRESHOLDS.HIGH) return 'text-green-400';
    if (confidence >= CONFIG.CONFIDENCE_THRESHOLDS.MEDIUM) return 'text-yellow-400';
    return 'text-red-400';
  };

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const totalPages = Math.ceil(pagination.totalItems / pagination.itemsPerPage);

  return (
    <div className="p-6 space-y-6">
      {/* Header Section */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h2 className="text-2xl font-extrabold mb-2 uppercase tracking-wide">
            <span className="bg-gradient-to-r from-green-400 to-emerald-500 bg-clip-text text-transparent">
              SCAN
            </span>
            <span className="bg-gradient-to-r from-yellow-300 to-amber-400 bg-clip-text text-transparent">
              HISTORY
            </span>
          </h2>
          <p className="text-gray-400">Track your contamination detection results over time</p>
        </div>
        
        {/* Export Controls */}
        <div className="flex items-center space-x-2 mt-4 lg:mt-0">
          <select
            className="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:border-blue-500"
            onChange={(e) => {
              const value = e.target.value as typeof CONFIG.EXPORT_FORMATS[number];
              if (CONFIG.EXPORT_FORMATS.includes(value)) {
                handleExport(value);
              }
            }}
            value=""
          >
            <option value="">Export as...</option>
            {CONFIG.EXPORT_FORMATS.map(format => (
              <option key={format} value={format}>{format.toUpperCase()}</option>
            ))}
          </select>
          <button
            onClick={() => handleExport('csv')}
            disabled={isLoading}
            className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm transition-colors disabled:opacity-50"
          >
            <Download className="w-4 h-4" />
            <span>Export</span>
          </button>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-blue-600/20 rounded-lg flex items-center justify-center">
              <Calendar className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-blue-400">{statistics.total}</p>
              <p className="text-gray-400 text-sm">Total Scans</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-green-600/20 rounded-lg flex items-center justify-center">
              <CheckCircle className="w-6 h-6 text-green-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-green-400">{statistics.healthy}</p>
              <p className="text-gray-400 text-sm">Healthy Bags</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-red-600/20 rounded-lg flex items-center justify-center">
              <AlertTriangle className="w-6 h-6 text-red-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-red-400">{statistics.contaminated}</p>
              <p className="text-gray-400 text-sm">Contaminated</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-purple-600/20 rounded-lg flex items-center justify-center">
              <span className="text-purple-400 font-bold">%</span>
            </div>
            <div>
              <p className="text-2xl font-bold text-purple-400">{statistics.avgConfidence}</p>
              <p className="text-gray-400 text-sm">Avg Confidence</p>
            </div>
          </div>
        </div>
      </div>

      {/* Advanced Filters */}
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input
              type="text"
              placeholder="Search bags, contamination..."
              value={filters.searchTerm}
              onChange={(e) => handleFilterChange('searchTerm', e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 text-white pl-10 pr-4 py-2 rounded-lg text-sm focus:outline-none focus:border-blue-500"
            />
          </div>

          {/* Status Filter */}
          <select
            value={filters.status}
            onChange={(e) => handleFilterChange('status', e.target.value as FilterOptions['status'])}
            className="bg-gray-700 border border-gray-600 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
          >
            <option value="all">All Status</option>
            <option value="healthy">Healthy</option>
            <option value="contaminated">Contaminated</option>
          </select>

          {/* Date Range Filter */}
          <select
            value={filters.dateRange}
            onChange={(e) => handleFilterChange('dateRange', e.target.value as FilterOptions['dateRange'])}
            className="bg-gray-700 border border-gray-600 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
          >
            <option value="all">All Time</option>
            <option value="today">Today</option>
            <option value="week">Last Week</option>
            <option value="month">Last Month</option>
          </select>

          {/* Confidence Filter */}
          <div className="flex items-center space-x-2">
            <span className="text-gray-400 text-sm whitespace-nowrap">Min Confidence:</span>
            <input
              type="range"
              min="0"
              max="100"
              value={filters.minConfidence}
              onChange={(e) => handleFilterChange('minConfidence', parseInt(e.target.value))}
              className="flex-1"
            />
            <span className="text-gray-300 text-sm w-8">{filters.minConfidence}%</span>
          </div>
        </div>
      </div>

      {/* Bulk Actions */}
      {selectedLogs.size > 0 && (
        <div className="bg-blue-600/10 border border-blue-600/20 rounded-lg p-3">
          <div className="flex items-center justify-between">
            <span className="text-blue-400 text-sm">
              {selectedLogs.size} item{selectedLogs.size > 1 ? 's' : ''} selected
            </span>
            <div className="flex space-x-2">
              <button
                onClick={() => handleBulkAction('export', Array.from(selectedLogs))}
                className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm transition-colors"
              >
                Export Selected
              </button>
              <button
                onClick={() => handleBulkAction('delete', Array.from(selectedLogs))}
                className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded text-sm transition-colors"
              >
                Delete Selected
              </button>
            </div>
          </div>
        </div>
      )}

      {/* History Logs */}
      <div className="space-y-4">
        {isLoading ? (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mx-auto"></div>
            <p className="text-gray-400 mt-4">Processing...</p>
          </div>
        ) : paginatedData.length > 0 ? (
          paginatedData.map((log) => (
            <div key={log.id} className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
              <div className="p-4">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={selectedLogs.has(log.id)}
                      onChange={(e) => {
                        const newSelected = new Set(selectedLogs);
                        if (e.target.checked) {
                          newSelected.add(log.id);
                        } else {
                          newSelected.delete(log.id);
                        }
                        setSelectedLogs(newSelected);
                      }}
                      className="rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500 focus:ring-offset-0"
                    />
                    {getStatusIcon(log.status)}
                    <div>
                      <h3 className="text-white font-semibold">{log.bagId}</h3>
                      <div className="flex items-center space-x-4 text-sm text-gray-400">
                        <div className="flex items-center space-x-1">
                          <Clock className="w-4 h-4" />
                          <span>{formatDate(log.timestamp)}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors">
                      <Eye className="w-4 h-4" />
                    </button>
                    <button className="p-2 text-gray-400 hover:text-red-400 hover:bg-gray-700 rounded-lg transition-colors">
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
                  <div className="lg:col-span-3 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className={`px-3 py-1 rounded-full text-sm border ${getStatusColor(log.status)}`}>
                        {log.status.charAt(0).toUpperCase() + log.status.slice(1)}
                      </span>
                      <span className="text-sm">
                        <span className="text-gray-400">Confidence: </span>
                        <span className={`font-semibold ${getConfidenceColor(log.confidence)}`}>
                          {log.confidence}%
                        </span>
                      </span>
                    </div>

                    {log.contaminationType && (
                      <div>
                        <span className="text-gray-400 text-sm">Contamination: </span>
                        <span className="text-red-400 font-medium">{log.contaminationType}</span>
                      </div>
                    )}
                  </div>

                  <div className="flex justify-center lg:justify-end">
                    <div className="w-20 h-20 bg-gray-700 rounded-lg flex items-center justify-center">
                      <span className="text-gray-500 text-xs">IMG</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
              <Calendar className="w-8 h-8 text-gray-500" />
            </div>
            <h3 className="text-lg font-medium text-gray-400 mb-2">No scan history found</h3>
            <p className="text-gray-500 text-sm">
              {filters.status === 'all' && !filters.searchTerm
                ? 'Start scanning your mushroom bags to see history here'
                : 'Try adjusting your filters to see more results'}
            </p>
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex flex-col sm:flex-row items-center justify-between space-y-4 sm:space-y-0">
          <div className="flex items-center space-x-2">
            <span className="text-gray-400 text-sm">Show:</span>
            <select
              value={pagination.itemsPerPage}
              onChange={(e) => handleItemsPerPageChange(parseInt(e.target.value))}
              className="bg-gray-700 border border-gray-600 text-white rounded px-2 py-1 text-sm"
            >
              {CONFIG.ITEMS_PER_PAGE_OPTIONS.map(option => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
            <span className="text-gray-400 text-sm">
              of {pagination.totalItems} results
            </span>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => handlePageChange(pagination.currentPage - 1)}
              disabled={pagination.currentPage === 1}
              className="p-2 rounded-lg bg-gray-700 text-gray-300 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            
            <div className="flex space-x-1">
              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                const pageNum = i + Math.max(1, pagination.currentPage - 2);
                if (pageNum > totalPages) return null;
                
                return (
                  <button
                    key={pageNum}
                    onClick={() => handlePageChange(pageNum)}
                    className={`px-3 py-1 rounded text-sm transition-colors ${
                      pageNum === pagination.currentPage
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    {pageNum}
                  </button>
                );
              })}
            </div>

            <button
              onClick={() => handlePageChange(pagination.currentPage + 1)}
              disabled={pagination.currentPage === totalPages}
              className="p-2 rounded-lg bg-gray-700 text-gray-300 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Insights Section */}
      <div className="bg-gradient-to-r from-purple-600/10 to-blue-600/10 rounded-lg p-4 border border-purple-600/20">
        <h3 className="text-lg font-semibold text-white mb-3">📊 Analytics Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-purple-400 font-medium">Success Rate:</span>
            <p className="text-gray-300">
              {statistics.total > 0 ? Math.round((statistics.healthy / statistics.total) * 100) : 0}%
            </p>
          </div>
          <div>
            <span className="text-purple-400 font-medium">High Confidence:</span>
            <p className="text-gray-300">
              {processedData.filter(log => log.confidence >= CONFIG.CONFIDENCE_THRESHOLDS.HIGH).length} scans
            </p>
          </div>
          <div>
            <span className="text-purple-400 font-medium">This Week:</span>
            <p className="text-gray-300">
              {processedData.filter(log => 
                new Date(log.timestamp) > new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
              ).length} scans
            </p>
          </div>
          <div>
            <span className="text-purple-400 font-medium">Avg Confidence:</span>
            <p className="text-gray-300">{statistics.avgConfidence}%</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HistoryTab;