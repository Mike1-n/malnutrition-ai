-- Migration: Create user profiles table and trigger for approvals
-- Created at: 2026-06-15
-- Purpose: Manages user approval states and admin status for clinical staff.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS public.user_profiles (
  id UUID REFERENCES auth.users ON DELETE CASCADE PRIMARY KEY,
  email TEXT NOT NULL,
  is_approved BOOLEAN DEFAULT FALSE NOT NULL,
  is_admin BOOLEAN DEFAULT FALSE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Enable Row Level Security (RLS) on user_profiles
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if any
DROP POLICY IF EXISTS "Allow public select" ON public.user_profiles;
DROP POLICY IF EXISTS "Allow public insert" ON public.user_profiles;
DROP POLICY IF EXISTS "Allow public update" ON public.user_profiles;

-- Create policies for public access (since Streamlit app uses anon keys)
CREATE POLICY "Allow public select" ON public.user_profiles FOR SELECT TO public USING (true);
CREATE POLICY "Allow public insert" ON public.user_profiles FOR INSERT TO public WITH CHECK (true);
CREATE POLICY "Allow public update" ON public.user_profiles FOR UPDATE TO public USING (true) WITH CHECK (true);

-- Create a trigger function that automatically inserts a profile row when a new user signs up in auth.users
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.user_profiles (id, email, is_approved, is_admin)
  VALUES (
    new.id,
    new.email,
    -- Make the bootstrap administrator approved and admin automatically
    CASE WHEN new.email = 'admin@gmail.com' THEN TRUE ELSE FALSE END,
    CASE WHEN new.email = 'admin@gmail.com' THEN TRUE ELSE FALSE END
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Bind the trigger function to the auth.users table
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Clean up any existing default admin user to start fresh
-- ON DELETE CASCADE on auth.users will automatically clean up user_profiles
DELETE FROM auth.users WHERE email = 'admin@gmail.com';
DELETE FROM auth.identities WHERE id = 'a1a1a1a1-b2b2-c3c3-d4d4-e5e5e5e5e5e5';

-- Insert default admin user into auth.users
INSERT INTO auth.users (
  id,
  email,
  encrypted_password,
  email_confirmed_at,
  raw_app_meta_data,
  raw_user_meta_data,
  aud,
  role,
  created_at,
  updated_at
) VALUES (
  'a1a1a1a1-b2b2-c3c3-d4d4-e5e5e5e5e5e5',
  'admin@gmail.com',
  crypt('123456', gen_salt('bf', 10)),
  NOW(),
  '{"provider": "email", "providers": ["email"]}',
  '{}',
  'authenticated',
  'authenticated',
  NOW(),
  NOW()
);

-- Insert linked identity into auth.identities to enable login
INSERT INTO auth.identities (
  id,
  user_id,
  identity_data,
  provider,
  provider_id,
  last_sign_in_at,
  created_at,
  updated_at
) VALUES (
  'a1a1a1a1-b2b2-c3c3-d4d4-e5e5e5e5e5e5',
  'a1a1a1a1-b2b2-c3c3-d4d4-e5e5e5e5e5e5',
  '{"sub": "a1a1a1a1-b2b2-c3c3-d4d4-e5e5e5e5e5e5", "email": "admin@gmail.com"}'::jsonb,
  'email',
  'a1a1a1a1-b2b2-c3c3-d4d4-e5e5e5e5e5e5',
  NOW(),
  NOW(),
  NOW()
);

-- Ensure the user profile is explicitly created, approved, and admin
-- (Although the handle_new_user trigger will fire, we upsert to guarantee correctness)
INSERT INTO public.user_profiles (
  id,
  email,
  is_approved,
  is_admin,
  created_at
) VALUES (
  'a1a1a1a1-b2b2-c3c3-d4d4-e5e5e5e5e5e5',
  'admin@gmail.com',
  TRUE,
  TRUE,
  NOW()
)
ON CONFLICT (id) DO UPDATE SET
  is_approved = TRUE,
  is_admin = TRUE;

