# Singapore Air Quality Forecasting

English | [简体中文](README.zh-CN.md)

A project exploring air pollution forecasting under the theme of AI for Social Good.

## Goal

Use a sliding window of the most recent 24–48 hours of PM2.5 observations to predict hourly concentrations for the next 1–6 hours in Singapore, helping people plan travel and outdoor activities.

## Current Contents

- [Original NEA dataset](data/singapore-pm25-hourly-raw.csv): 109,584 hourly records covering Singapore's North, South, East, West, and Central regions, from April 2014 to the final timestamp at 00:00 on 1 October 2026.
- Dataset source, field descriptions, and initial quality checks in [English](data/README.md) and [Chinese](data/README.zh-CN.md).

The data has been downloaded. Data preprocessing, forecasting models, and the demonstration interface have not yet been implemented.

## Collaboration

Please fork this repository, create a `dev` branch in your fork, and submit changes through a Pull Request (PR) to this repository's `main` branch. Where possible, keep AI-generated instruction or constraint files, such as `AGENTS.md` and `CLAUDE.md`, local rather than committing them to the repository.
