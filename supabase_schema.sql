-- Supabase Database Schema for SEOExtract

-- Create enum for subscription plans
CREATE TYPE subscription_plan AS ENUM ('free', 'pro', 'premium');
CREATE TYPE payment_status AS ENUM ('pending', 'paid', 'failed', 'cancelled', 'refunded');
CREATE TYPE subscription_status AS ENUM ('active', 'inactive', 'cancelled', 'past_due', 'trialing');

-- Users table (extends Supabase auth.users)
CREATE TABLE public.user_profiles (
    id UUID REFERENCES auth.users(id) PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    plan subscription_plan DEFAULT 'free',
    subscription_status subscription_status DEFAULT 'inactive',
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    current_period_start TIMESTAMP WITH TIME ZONE,
    current_period_end TIMESTAMP WITH TIME ZONE,
    trial_end TIMESTAMP WITH TIME ZONE,
    auto_upgrade BOOLEAN DEFAULT true,
    plan_change_pending subscription_plan,
    plan_change_effective_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Payment history table
CREATE TABLE public.payments (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    stripe_payment_intent_id TEXT,
    amount INTEGER NOT NULL, -- Amount in cents
    currency TEXT DEFAULT 'usd',
    status payment_status DEFAULT 'pending',
    plan_purchased subscription_plan NOT NULL,
    payment_method TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Plan changes log table
CREATE TABLE public.plan_changes (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    from_plan subscription_plan,
    to_plan subscription_plan NOT NULL,
    change_reason TEXT, -- 'payment', 'upgrade', 'downgrade', 'admin', 'trial_end'
    changed_by UUID REFERENCES auth.users(id), -- Admin who made the change
    effective_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Search history table
CREATE TABLE public.search_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    keywords JSONB, -- Array of keywords found
    seo_suggestions JSONB, -- Array of SEO suggestions
    seo_data JSONB, -- Full SEO metadata
    keyword_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for faster queries
CREATE INDEX idx_search_history_user_id ON public.search_history(user_id);
CREATE INDEX idx_search_history_created_at ON public.search_history(created_at DESC);

-- Usage tracking table (enhanced)
CREATE TABLE public.user_usage (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    month TEXT NOT NULL, -- Format: 'YYYY-MM'
    audits_used INTEGER DEFAULT 0,
    keywords_generated INTEGER DEFAULT 0,
    exports_used INTEGER DEFAULT 0,
    api_calls_used INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, month)
);

-- Audit history table (enhanced)
CREATE TABLE public.audit_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    keywords JSONB,
    seo_suggestions JSONB,
    audit_results JSONB,
    keyword_count INTEGER,
    suggestion_count INTEGER,
    plan_at_time subscription_plan,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Admin actions log
CREATE TABLE public.admin_actions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    admin_id UUID REFERENCES auth.users(id),
    action_type TEXT NOT NULL, -- 'plan_change', 'user_suspend', 'refund', etc.
    target_user_id UUID REFERENCES public.user_profiles(id),
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Plan-specific user views
CREATE VIEW public.free_plan_users AS
SELECT * FROM public.user_profiles WHERE plan = 'free';

CREATE VIEW public.pro_plan_users AS
SELECT * FROM public.user_profiles WHERE plan = 'pro';

CREATE VIEW public.premium_plan_users AS
SELECT * FROM public.user_profiles WHERE plan = 'premium';

-- Active subscribers view
CREATE VIEW public.active_subscribers AS
SELECT * FROM public.user_profiles 
WHERE plan IN ('pro', 'premium') 
AND subscription_status = 'active';

-- Users with pending plan changes
CREATE VIEW public.pending_plan_changes AS
SELECT * FROM public.user_profiles 
WHERE plan_change_pending IS NOT NULL 
AND plan_change_effective_date <= NOW();

-- RLS (Row Level Security) policies
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.payments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.plan_changes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.admin_actions ENABLE ROW LEVEL SECURITY;

-- Users can only see their own data
CREATE POLICY "Users can view own profile" ON public.user_profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.user_profiles
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can view own usage" ON public.user_usage
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can view own audit history" ON public.audit_history
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can view own payments" ON public.payments
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can view own plan changes" ON public.plan_changes
    FOR SELECT USING (auth.uid() = user_id);

-- Admin policies (admin user can see everything)
CREATE POLICY "Admin can view all profiles" ON public.user_profiles
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM public.user_profiles 
            WHERE id = auth.uid() 
            AND email = 'tall3aadam@gmail.com'
        )
    );

CREATE POLICY "Admin can view all admin actions" ON public.admin_actions
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM public.user_profiles 
            WHERE id = auth.uid() 
            AND email = 'tall3aadam@gmail.com'
        )
    );

-- Functions to handle user creation
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_profiles (id, email, full_name)
    VALUES (NEW.id, NEW.email, NEW.raw_user_meta_data->>'full_name');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to upgrade user plan after payment
CREATE OR REPLACE FUNCTION public.upgrade_user_plan(
    user_email TEXT,
    new_plan subscription_plan,
    stripe_subscription_id TEXT DEFAULT NULL,
    period_end TIMESTAMP WITH TIME ZONE DEFAULT NULL
)
RETURNS BOOLEAN AS $$
DECLARE
    user_record RECORD;
    old_plan subscription_plan;
BEGIN
    -- Get user record
    SELECT * INTO user_record FROM public.user_profiles WHERE email = user_email;

    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;

    old_plan := user_record.plan;

    -- Update user plan
    UPDATE public.user_profiles 
    SET 
        plan = new_plan,
        subscription_status = 'active',
        stripe_subscription_id = COALESCE(upgrade_user_plan.stripe_subscription_id, stripe_subscription_id),
        current_period_start = NOW(),
        current_period_end = COALESCE(period_end, NOW() + INTERVAL '1 month'),
        updated_at = NOW()
    WHERE email = user_email;

    -- Log the plan change
    INSERT INTO public.plan_changes (user_id, from_plan, to_plan, change_reason)
    VALUES (user_record.id, old_plan, new_plan, 'payment');

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to process pending plan changes
CREATE OR REPLACE FUNCTION public.process_pending_plan_changes()
RETURNS INTEGER AS $$
DECLARE
    changes_processed INTEGER := 0;
    user_record RECORD;
BEGIN
    FOR user_record IN 
        SELECT * FROM public.user_profiles 
        WHERE plan_change_pending IS NOT NULL 
        AND plan_change_effective_date <= NOW()
    LOOP
        -- Log the change
        INSERT INTO public.plan_changes (user_id, from_plan, to_plan, change_reason)
        VALUES (user_record.id, user_record.plan, user_record.plan_change_pending, 'scheduled');

        -- Apply the change
        UPDATE public.user_profiles 
        SET 
            plan = plan_change_pending,
            plan_change_pending = NULL,
            plan_change_effective_date = NULL,
            updated_at = NOW()
        WHERE id = user_record.id;

        changes_processed := changes_processed + 1;
    END LOOP;

    RETURN changes_processed;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to handle subscription cancellation
CREATE OR REPLACE FUNCTION public.cancel_subscription(
    user_email TEXT,
    immediate BOOLEAN DEFAULT FALSE
)
RETURNS BOOLEAN AS $$
DECLARE
    user_record RECORD;
BEGIN
    SELECT * INTO user_record FROM public.user_profiles WHERE email = user_email;

    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;

    IF immediate THEN
        -- Immediate downgrade to free
        UPDATE public.user_profiles 
        SET 
            plan = 'free',
            subscription_status = 'cancelled',
            updated_at = NOW()
        WHERE email = user_email;

        INSERT INTO public.plan_changes (user_id, from_plan, to_plan, change_reason)
        VALUES (user_record.id, user_record.plan, 'free', 'cancellation');
    ELSE
        -- Schedule downgrade at period end
        UPDATE public.user_profiles 
        SET 
            subscription_status = 'cancelled',
            plan_change_pending = 'free',
            plan_change_effective_date = current_period_end,
            updated_at = NOW()
        WHERE email = user_email;
    END IF;

    RETURN TRUE;
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
INSERT INTO public.user_profiles (id, email, full_name, plan, subscription_status)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'tall3aadam@gmail.com',
    'Admin User',
    'premium',
    'active'
) ON CONFLICT (email) DO UPDATE SET 
    plan = 'premium',
    subscription_status = 'active';

-- Sample data for testing
INSERT INTO public.user_profiles (id, email, full_name, plan, subscription_status) VALUES
    ('00000000-0000-0000-0000-000000000002', 'free.user@example.com', 'Free User', 'free', 'inactive'),
    ('00000000-0000-0000-0000-000000000003', 'pro.user@example.com', 'Pro User', 'pro', 'active'),
    ('00000000-0000-0000-0000-000000000004', 'premium.user@example.com', 'Premium User', 'premium', 'active')
ON CONFLICT (email) DO NOTHING;

-- Indexes for better performance
CREATE INDEX idx_user_profiles_email ON public.user_profiles(email);
CREATE INDEX idx_user_profiles_plan ON public.user_profiles(plan);
CREATE INDEX idx_user_profiles_subscription_status ON public.user_profiles(subscription_status);
CREATE INDEX idx_payments_user_id ON public.payments(user_id);
CREATE INDEX idx_payments_status ON public.payments(status);
CREATE INDEX idx_plan_changes_user_id ON public.plan_changes(user_id);
CREATE INDEX idx_user_usage_user_month ON public.user_usage(user_id, month);
CREATE INDEX idx_audit_history_user_id ON public.audit_history(user_id);
ALTER TABLE public.user_profiles ADD COLUMN api_key TEXT UNIQUE;