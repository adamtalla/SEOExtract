
-- Supabase Database Schema for SEOExtract

-- Create enum for subscription plans
CREATE TYPE subscription_plan AS ENUM ('free', 'pro', 'premium');

-- Users table (extends Supabase auth.users)
CREATE TABLE public.user_profiles (
    id UUID REFERENCES auth.users(id) PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    plan subscription_plan DEFAULT 'free',
    stripe_customer_id TEXT,
    subscription_id TEXT,
    subscription_status TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Usage tracking table
CREATE TABLE public.user_usage (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    month TEXT NOT NULL, -- Format: 'YYYY-MM'
    audits_used INTEGER DEFAULT 0,
    keywords_generated INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, month)
);

-- Audit history table
CREATE TABLE public.audit_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    keywords JSONB,
    seo_suggestions JSONB,
    keyword_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Plan-specific user views
CREATE VIEW public.free_plan_users AS
SELECT * FROM public.user_profiles WHERE plan = 'free';

CREATE VIEW public.pro_plan_users AS
SELECT * FROM public.user_profiles WHERE plan = 'pro';

CREATE VIEW public.premium_plan_users AS
SELECT * FROM public.user_profiles WHERE plan = 'premium';

-- RLS (Row Level Security) policies
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_history ENABLE ROW LEVEL SECURITY;

-- Users can only see their own data
CREATE POLICY "Users can view own profile" ON public.user_profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.user_profiles
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can view own usage" ON public.user_usage
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can view own audit history" ON public.audit_history
    FOR ALL USING (auth.uid() = user_id);

-- Functions to handle user creation
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_profiles (id, email, full_name)
    VALUES (NEW.id, NEW.email, NEW.raw_user_meta_data->>'full_name');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to create profile on signup
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Function to update timestamps
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_user_profiles_updated_at
    BEFORE UPDATE ON public.user_profiles
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_user_usage_updated_at
    BEFORE UPDATE ON public.user_usage
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- Insert admin user with premium access
INSERT INTO public.user_profiles (id, email, full_name, plan)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'tall3aadam@gmail.com',
    'Admin User',
    'premium'
) ON CONFLICT (email) DO UPDATE SET plan = 'premium';

-- Sample data for testing
INSERT INTO public.user_profiles (id, email, full_name, plan) VALUES
    ('00000000-0000-0000-0000-000000000002', 'free.user@example.com', 'Free User', 'free'),
    ('00000000-0000-0000-0000-000000000003', 'pro.user@example.com', 'Pro User', 'pro'),
    ('00000000-0000-0000-0000-000000000004', 'premium.user@example.com', 'Premium User', 'premium')
ON CONFLICT (email) DO NOTHING;
