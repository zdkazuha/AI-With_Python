import sql from "mssql";
import dotenv from "dotenv";

dotenv.config();

const config = {
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    server: process.env.DB_SERVER,
    database: process.env.DB_DATABASE,
    port: Number(process.env.DB_PORT),

    options: {
        trustServerCertificate: true
    }
};

export async function getConnection() {

    try {

        return await sql.connect(config);

    } catch (error) {

        console.error("Database connection error:");

        console.error(error.message);

        throw error;
    }
}