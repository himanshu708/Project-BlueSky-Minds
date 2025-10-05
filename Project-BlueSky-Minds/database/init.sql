-- Initialize WeatherOdds Pro Database

-- Create locations table
CREATE TABLE IF NOT EXISTS locations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    country VARCHAR(100),
    state VARCHAR(100),
    city VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(latitude, longitude)
);

-- Create weather_cache table for caching results
CREATE TABLE IF NOT EXISTS weather_cache (
    id SERIAL PRIMARY KEY,
    cache_key VARCHAR(255) UNIQUE NOT NULL,
    location_id INTEGER REFERENCES locations(id),
    date_requested DATE,
    variables TEXT[],
    result_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    hit_count INTEGER DEFAULT 0
);

-- Create historical_data table for storing processed NASA data
CREATE TABLE IF NOT EXISTS historical_data (
    id SERIAL PRIMARY KEY,
    location_id INTEGER REFERENCES locations(id),
    variable VARCHAR(50) NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    value DECIMAL(10, 4),
    data_source VARCHAR(100) DEFAULT 'NASA_MERRA2',
    quality_flag INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(location_id, variable, year, month)
);

-- Create probability_results table for storing calculated probabilities
CREATE TABLE IF NOT EXISTS probability_results (
    id SERIAL PRIMARY KEY,
    location_id INTEGER REFERENCES locations(id),
    variable VARCHAR(50) NOT NULL,
    target_date DATE,
    mean_value DECIMAL(10, 4),
    std_deviation DECIMAL(10, 4),
    percentiles JSONB,
    extreme_probabilities JSONB,
    trend_analysis JSONB,
    confidence_score DECIMAL(3, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create forecast_results table for extended forecasts
CREATE TABLE IF NOT EXISTS forecast_results (
    id SERIAL PRIMARY KEY,
    location_id INTEGER REFERENCES locations(id),
    variable VARCHAR(50) NOT NULL,
    forecast_month INTEGER NOT NULL,
    forecast_year INTEGER NOT NULL,
    predicted_value DECIMAL(10, 4),
    confidence_interval JSONB,
    probability_extremes JSONB,
    methodology VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_locations_coords ON locations(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_weather_cache_key ON weather_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_weather_cache_expires ON weather_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_historical_data_location_var ON historical_data(location_id, variable);
CREATE INDEX IF NOT EXISTS idx_historical_data_date ON historical_data(year, month);
CREATE INDEX IF NOT EXISTS idx_probability_results_location ON probability_results(location_id, target_date);
CREATE INDEX IF NOT EXISTS idx_forecast_results_location ON forecast_results(location_id, forecast_year, forecast_month);

-- Insert some sample locations for testing
INSERT INTO locations (name, latitude, longitude, country, state, city) VALUES
('New York City', 40.7128, -74.0060, 'USA', 'New York', 'New York'),
('Los Angeles', 34.0522, -118.2437, 'USA', 'California', 'Los Angeles'),
('London', 51.5074, -0.1278, 'UK', 'England', 'London'),
('Tokyo', 35.6762, 139.6503, 'Japan', 'Tokyo', 'Tokyo'),
('Sydney', -33.8688, 151.2093, 'Australia', 'New South Wales', 'Sydney')
ON CONFLICT (latitude, longitude) DO NOTHING;

-- Create a function to clean expired cache entries
CREATE OR REPLACE FUNCTION clean_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM weather_cache WHERE expires_at < CURRENT_TIMESTAMP;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create a function to get or create location
CREATE OR REPLACE FUNCTION get_or_create_location(
    p_latitude DECIMAL(10, 8),
    p_longitude DECIMAL(11, 8),
    p_name VARCHAR(255) DEFAULT NULL,
    p_country VARCHAR(100) DEFAULT NULL,
    p_state VARCHAR(100) DEFAULT NULL,
    p_city VARCHAR(100) DEFAULT NULL
)
RETURNS INTEGER AS $$
DECLARE
    location_id INTEGER;
BEGIN
    -- Try to find existing location
    SELECT id INTO location_id
    FROM locations
    WHERE latitude = p_latitude AND longitude = p_longitude;
    
    -- If not found, create new location
    IF location_id IS NULL THEN
        INSERT INTO locations (name, latitude, longitude, country, state, city)
        VALUES (p_name, p_latitude, p_longitude, p_country, p_state, p_city)
        RETURNING id INTO location_id;
    END IF;
    
    RETURN location_id;
END;
$$ LANGUAGE plpgsql;