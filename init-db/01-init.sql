-- Panoptique Ferroviaire - Database Initialization
-- PostGIS spatial database with pgrouting extension

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Enable pgrouting extension for network analysis
CREATE EXTENSION IF NOT EXISTS pgrouting;

-- Create schema for railway data
CREATE SCHEMA IF NOT EXISTS railway;

-- Table: Train entities with current state
CREATE TABLE IF NOT EXISTS railway.trains (
    train_id VARCHAR(100) PRIMARY KEY,
    trip_id VARCHAR(100),
    route_id VARCHAR(100),
    operator_name VARCHAR(200),
    
    -- Spatial data
    position GEOMETRY(POINT, 4326),
    bearing DOUBLE PRECISION,
    
    -- Kinematic data
    velocity DOUBLE PRECISION,
    acceleration DOUBLE PRECISION,
    
    -- Metadata
    last_update TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    state VARCHAR(50),
    
    -- Indexes
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table: Historical train positions
CREATE TABLE IF NOT EXISTS railway.train_history (
    id BIGSERIAL PRIMARY KEY,
    train_id VARCHAR(100) NOT NULL,
    position GEOMETRY(POINT, 4326),
    velocity DOUBLE PRECISION,
    acceleration DOUBLE PRECISION,
    bearing DOUBLE PRECISION,
    state VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table: Railway track network (for future use with pgrouting)
CREATE TABLE IF NOT EXISTS railway.tracks (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200),
    route_id VARCHAR(100),
    geom GEOMETRY(LINESTRING, 4326),
    length DOUBLE PRECISION,
    max_speed DOUBLE PRECISION,
    gradient DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table: Operators and their resources
CREATE TABLE IF NOT EXISTS railway.operators (
    id SERIAL PRIMARY KEY,
    operator_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    resource_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table: GTFS-RT data sources
CREATE TABLE IF NOT EXISTS railway.data_sources (
    id SERIAL PRIMARY KEY,
    operator_id VARCHAR(100) REFERENCES railway.operators(operator_id),
    url TEXT NOT NULL,
    resource_type VARCHAR(50),
    title TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    last_successful_harvest TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create spatial index on trains position
CREATE INDEX IF NOT EXISTS idx_trains_position ON railway.trains USING GIST(position);

-- Create index on train_id for history
CREATE INDEX IF NOT EXISTS idx_train_history_train_id ON railway.train_history(train_id);
CREATE INDEX IF NOT EXISTS idx_train_history_timestamp ON railway.train_history(timestamp DESC);

-- Create spatial index on train history
CREATE INDEX IF NOT EXISTS idx_train_history_position ON railway.train_history USING GIST(position);

-- Create spatial index on tracks
CREATE INDEX IF NOT EXISTS idx_tracks_geom ON railway.tracks USING GIST(geom);

-- Create index on route_id
CREATE INDEX IF NOT EXISTS idx_trains_route ON railway.trains(route_id);
CREATE INDEX IF NOT EXISTS idx_tracks_route ON railway.tracks(route_id);

-- Function: Update updated_at timestamp
CREATE OR REPLACE FUNCTION railway.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: Auto-update updated_at on trains table
CREATE TRIGGER update_trains_updated_at BEFORE UPDATE ON railway.trains
    FOR EACH ROW EXECUTE FUNCTION railway.update_updated_at_column();

-- Grant permissions
GRANT USAGE ON SCHEMA railway TO panoptique;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA railway TO panoptique;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA railway TO panoptique;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Panoptique Ferroviaire database initialized successfully';
END $$;
