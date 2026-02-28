const express = require("express");
const bodyParser = require("body-parser");
const bcrypt = require("bcryptjs");
const { Pool } = require("pg");
const path = require("path");

const app = express();
const port = 3000;

// Middleware to parse incoming JSON data from the frontend
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

// --- THIS IS THE LINK TO YOUR HTML ---
// When someone visits http://localhost:3000, serve the index.html file
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});



// PostgreSQL connection
const pool = new Pool({
  user: "postgres",
  host: "localhost",
  database: "mytask",
  password: "223202112",
  port: 5432,
});

// Auto-create users table
async function initDB() {
  try {
    await pool.query(`
      CREATE TABLE IF NOT EXISTS user2 (
        id SERIAL PRIMARY KEY,
        username VARCHAR(200),
        email VARCHAR(200) UNIQUE,
        password VARCHAR(355)
      );
    `);
    console.log("✅ Database table 'user2' is ready");
  } catch (err) {
    console.error("❌ Table creation error:", err.message);
  }
}

// Signup route
app.post("/signup", async (req, res) => {
  const { username, email, password } = req.body;
  try {
    const hashedPassword = await bcrypt.hash(password, 10);
    const result = await pool.query(
      "INSERT INTO user2 (username, email, password) VALUES ($1, $2, $3) RETURNING *",
      [username, email, hashedPassword]
    );
    res.status(201).json({ message: "User registered successfully!" });
  } catch (err) {
    // Check if the error is because the email already exists in the database
    if (err.code === '23505') {
      return res.status(400).json({ error: "Email already exists. Try logging in." });
    }
    console.error("Signup error:", err.message);
    res.status(500).json({ error: "Database error" });
  }
});

// Login route
app.post("/login", async (req, res) => {
  const { email, password } = req.body;
  try {
    const result = await pool.query("SELECT * FROM user2 WHERE email=$1", [email]);
    if (result.rows.length === 0) return res.status(401).json({ error: "Invalid email or password" });

    const user = result.rows[0];
    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword) return res.status(401).json({ error: "Invalid email or password" });

    // Send the username back to the frontend
    res.json({
      message: "Login successful!",
      username: user.username
    });
  } catch (err) {
    res.status(500).json({ error: "Database error" });
  }
});

// Start server after DB connection
(async () => {
  try {
    console.log("Connecting to PostgreSQL...");
    await pool.connect();
    console.log("✅ Connected to PostgreSQL");
    await initDB();
    app.listen(port, () => console.log(`🚀 Server running at http://localhost:${port}`));
  } catch (err) {
    console.error("❌ Fatal DB connection error:", err.message);
    console.error("Ensure PostgreSQL is running and your credentials are correct.");
    process.exit(1);
  }
})();