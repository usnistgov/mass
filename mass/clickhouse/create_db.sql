-- Create a Dastard output database.
-- This is just a crude start (Feb 23, 2024)

-- To run from terminal:
-- clickhouse client --queries-file create_tables.sql

CREATE TABLE IF NOT EXISTS pulses (
    `channel_id`      FixedString(26) Comment 'A ULID. Can be joined to channels.id',
    -- `timestamp`       DateTime64(9) Comment 'PC clock time at the moment of the trigger. Might want to skip this?',
    `subframe_count`  Int64 Comment 'Subframe counts (since Dastard started) at the moment of the trigger',
    `pulse`           String Comment 'Pulse data record (decode as uint16)',
)
    ENGINE = MergeTree()
    PRIMARY KEY (channel_id)
    ORDER BY (channel_id, subframe_count)
    COMMENT 'Each row is a single pulse record from one channel';

CREATE TABLE IF NOT EXISTS channels (
    `id`                 FixedString(26) PRIMARY KEY Comment 'A ULID for this channel',
    `datarun_id`         FixedString(26)  Comment 'A ULID. Can be joined to dataruns.id',
    `channel_number`     UInt32,
    `channel_group`      LowCardinality(String) Default '',
    `row_number`         UInt32 Default 0 Comment 'TDM row number, or 0 for µMUX',
    `column_number`      UInt32 Default 0 Comment 'TDM column number, or 0 for µMUX',
    `tone_frequency`     Float64 Default 0.0 Comment 'µMUX tone frequency in GHz, or 0 for TDM systems',
    `subframe_divisions` UInt32,
    `subframe_offset`    UInt32 Default 0,
    `presamples`         UInt32,
    `total_samples`      UInt32,
    `first_record_time`  DateTime64(9),
)
    ENGINE = MergeTree()
    ORDER BY (id)
    COMMENT 'Each row is a single microcalorimeter sensor in one data run';

CREATE TABLE IF NOT EXISTS dataruns (
    `id`              FixedString(26) PRIMARY KEY Comment 'A ULID for this run',
    `date_run_code`   String,
    `intention`       LowCardinality(String) DEFAULT 'unknown',
    `creator`         LowCardinality(String) DEFAULT 'unknown',
    `datasource`      LowCardinality(String) DEFAULT 'unknown',
    `daq_version`     LowCardinality(String) DEFAULT 'unknown',
    `daq_githash`     LowCardinality(String) DEFAULT 'unknown',
    `number_rows`     Int32 Default -1 Comment 'Number of TDM rows, or -1 for µMUX',
    `number_columns`  Int32 Default -1 Comment 'Number of TDM columns, or -1 for µMUX',
    `number_channels` UInt32,
    `timebase`        Float64,
    `server_start`    DateTime64(6),
)
    ENGINE = MergeTree()
    COMMENT 'Each row is a data run, with multiple microcalorimeter sensors running in parallel';

CREATE TABLE IF NOT EXISTS external_triggers (
    `datarun_id`      FixedString(26) Comment 'A ULID. Can be joined to dataruns.id',
    `subframe_count`  Int64 Comment 'Subframe counts (since Dastard started) at the moment of the external trigger',
)
    ENGINE = MergeTree()
    ORDER BY (datarun_id, subframe_count)
    COMMENT 'Each row is an instance of the external trigger';

CREATE TABLE IF NOT EXISTS experiment_state (
    `datarun_id`      FixedString(26) Comment 'A ULID. Can be joined to dataruns.id',
    `state`           String,
    `subframe_count`  Int64 Comment 'Subframe counts (since Dastard started) at the moment the new state was registered',
)
    ENGINE = MergeTree()
    ORDER BY (datarun_id, subframe_count)
    COMMENT 'Each row is an instance of the experiment state changing';
