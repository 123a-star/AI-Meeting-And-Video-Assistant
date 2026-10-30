
import os
import streamlit as st
from supabase import create_client


def get_supabase_client():
    """Create one Supabase client per Streamlit user session."""

    if "supabase_client" not in st.session_state:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise ValueError("Supabase credentials are missing.")

        st.session_state.supabase_client = create_client(url, key)

    return st.session_state.supabase_client


def sign_up(email, password):
    client = get_supabase_client()
    return client.auth.sign_up({
        "email": email,
        "password": password
    })


def sign_in(email, password):
    client = get_supabase_client()
    return client.auth.sign_in_with_password({
        "email": email,
        "password": password
    })


def sign_out():
    client = get_supabase_client()
    client.auth.sign_out()
    st.session_state.pop("authenticated_user", None)


def get_current_user():
    return st.session_state.get("authenticated_user")
