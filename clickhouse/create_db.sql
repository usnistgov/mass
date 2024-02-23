-- Create a Dastard output database.
-- This is just a crude start (Feb 23, 2024)

-- To run from terminal:
-- clickhouse client --queries-file create_tables.sql

CREATE TABLE IF NOT EXISTS pulses (
    `datarun_id`      UInt32  Comment 'Can be joined to dataruns.id',
    `channel_number`  UInt32,
    -- `timestamp`       DateTime64(9) Comment 'PC clock time at the moment of the trigger. Might want to skip this?',
    `subframe_count`  Int64 Comment 'Subframe counts (since Dastard started) at the moment of the trigger',
    `pulse`           Array(UInt16) Comment 'Pulse data record',
) 
    ENGINE = MergeTree()
    PRIMARY KEY (datarun_id, channel_number)
    ORDER BY (datarun_id, channel_number, subframe_count)
    COMMENT 'Each row is a single pulse record from one channel';

CREATE TABLE IF NOT EXISTS channels (
    `datarun_id`         UInt32  Comment 'Can be joined to dataruns.id',
    `channel_number`     UInt32,
    `channel_group`      LowCardinality(String) Default '',
    `row_number`         UInt32 NULL Comment 'TDM row number, or NULL for µMUX',
    `column_number`      UInt32 NULL Comment 'TDM column number, or NULL for µMUX',
    `tone_frequency`     Float64 NULL Comment 'µMUX tone frequency in GHz, or NULL for TDM systems',
    `subframe_divisions` UInt32,
    `subframe_offset`    UInt32 DEFAULT 0,
    `presamples`         UInt32,
    `total_samples`      UInt32,
    `first_record_time`  DateTime64(9),
)
    ENGINE = MergeTree()
    ORDER BY (datarun_id, channel_number)
    COMMENT 'Each row is a single microcalorimeter sensor in one data run';

CREATE TABLE IF NOT EXISTS dataruns (
    `id`              UInt32 PRIMARY KEY,
    `date_run_code`   String,
    `intention`       LowCardinality(String) DEFAULT 'unknown',
    `creator`         LowCardinality(String) DEFAULT 'unknown',
    `datasource`      LowCardinality(String) DEFAULT 'unknown',
    `daq_version`     LowCardinality(String) DEFAULT 'unknown',
    `daq_githash`     LowCardinality(String) DEFAULT 'unknown',
    `number_rows`     UInt32 NULL Comment 'Number of TDM rows, or NULL for µMUX',
    `number_columns`  UInt32 NULL Comment 'Number of TDM columns, or NULL for µMUX',
    `number_channels` UInt32,
    `timebase`        Float64,
    `server_start`    DateTime64(6),
)
    ENGINE = MergeTree()
    COMMENT 'Each row is a data run, with multiple microcalorimeter sensors running in parallel';

CREATE TABLE IF NOT EXISTS external_triggers (
    `datarun_id`      UInt32 Comment 'Can be joined to dataruns.id',
    `subframe_count`  Int64 Comment 'Subframe counts (since Dastard started) at the moment of the external trigger',
)
    ENGINE = MergeTree()
    ORDER BY (datarun_id, subframe_count)
    COMMENT 'Each row is an instance of the external trigger';

CREATE TABLE IF NOT EXISTS experiment_state (
    `datarun_id`      UInt32 Comment 'Can be joined to dataruns.id',
    `state`           String,
    `subframe_count`  Int64 Comment 'Subframe counts (since Dastard started) at the moment the new state was registered',
)
    ENGINE = MergeTree()
    ORDER BY (datarun_id, subframe_count)
    COMMENT 'Each row is an instance of the experiment state changing';
