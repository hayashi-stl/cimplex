//! Helps hide private details from public traits.
//! This module is private.

pub enum Key {}

pub trait Lock {}

impl Lock for Key {}
