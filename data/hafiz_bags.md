# Hafiz Bags — Factory Management App

## Tech Stack
React Native (Expo), Node.js, Express, MongoDB Atlas

## Overview
A factory management mobile application built for a family-owned bag manufacturing business, replacing manual/paper-based processes with a mobile-first system. Originally built with a workers portal module; independently rebuilt from scratch with the same core idea and approach, but with the workers portal removed and replaced by a new Workers Activity module for tracking worker activity/output.

## Technical Challenges Solved
- Diagnosed and resolved MongoDB Atlas SRV DNS resolution failures caused by local ISP-level DNS blocking (a known issue for developers connecting to MongoDB Atlas from Pakistan), implementing a workaround to maintain a stable database connection.
- Built the mobile client with Expo for faster iteration and cross-platform (Android/iOS) support without native build overhead.
- Designed a Node/Express REST API backend to serve as the bridge between the mobile app and MongoDB Atlas.

## Rebuild Notes
The app has since been rebuilt independently (same concept and approach as the original), with the workers portal module removed and a new Workers Activity module added in its place, to better track worker-level activity within the factory.

## Why This Matters For Client Work
This is real production software built for a real operating business — not a tutorial project or a spec/demo site. It demonstrates full-stack mobile development (React Native), backend API design (Node/Express), and cloud database work (MongoDB Atlas), including debugging real infrastructure problems rather than following a happy-path tutorial.

## Relevant For Job Posts Mentioning
React Native, Expo, mobile app development, Node.js backend, Express API, MongoDB, business/inventory/operations management apps, full-stack mobile development.
