# Silva's Detailing — Booking & Service Management Site

## Project Name
Silva's Detailing

## Problem Solved
Auto detailing businesses typically rely on phone calls and walk-ins for booking, leading to scheduling conflicts, missed appointments, and no-shows. Silva's Detailing is a professional booking website that lets customers browse service packages, select add-ons, choose available time slots, and pay deposits online — turning a chaotic manual process into a streamlined digital experience.

## Tech Stack
- **Frontend**: Next.js (React), Tailwind CSS, Framer Motion for animations
- **Backend**: Next.js API Routes, serverless functions
- **Database**: PostgreSQL via Prisma ORM
- **Calendar**: Google Calendar API sync for real-time availability
- **Hosting**: Vercel with automatic CI/CD
- **Notifications**: Twilio SMS for appointment reminders

## Key Features
- **Service Package Browser**: Visual cards showing Basic Wash, Full Detail, Ceramic Coating, etc., with pricing and time estimates
- **Real-Time Booking Calendar**: Interactive calendar showing available slots; prevents double-booking via Google Calendar sync
- **Customer Portal**: Returning customers can view booking history, reschedule, or cancel within policy window
- **Admin Dashboard**: Owner can manage services, pricing, availability windows, and view revenue analytics
- **Before/After Gallery**: Photo gallery with slider comparisons showcasing completed work
- **SMS Reminders**: Automated Twilio notifications 24h and 2h before appointments

## Project Context
This was built as a spec/demo project to showcase a complete, production-quality booking system architecture — not a live deployment with measured business metrics. It demonstrates end-to-end capability across payment processing, third-party calendar sync, and automated customer communication, designed around the real workflow problems auto detailing and service businesses face (phone-based booking, double-booking, no-shows).

## Technical Strengths Demonstrated
- Full payment integration (Stripe) including deposit-only and full-payment flows
- Third-party API orchestration under real constraints (Google Calendar availability sync, Twilio SMS scheduling)
- Responsive, mobile-first UI design suited to service-business customers booking on the go
- Admin/owner-facing CRUD tooling built for non-technical daily use, not just the customer-facing side