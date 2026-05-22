import express from "express";
import { getConnection } from "./db.js";

const app = express();

app.use(express.json());

/*
    MCP Tool:
    get_products(category)
*/

app.get("/products", async (req, res) => {

    try {

        const db = await getConnection();

        const result = await db.request()
            .query(`
                SELECT *    
                FROM Products
            `);

        res.json({
            tool: "get_products",
            count: result.recordset.length,
            products: result.recordset
        });

    } catch (error) {

        console.error(error);

        res.status(500).json({
            error: error.message
        });
  }
});

app.get("/categories", async (req, res) => {

    try {

        const db = await getConnection();

        const result = await db.request()
            .query(`
                SELECT Id, Name 
                FROM Categories
            `);

        res.json({
            tool: "get_categories",
            count: result.recordset.length,
            categories: result.recordset
        });

    } catch (error) {

        console.error(error);

        res.status(500).json({
            error: error.message
        });
  } 
});

app.get("/products/:categoryId", async (req, res) => {

    try {

        const categoryId = req.params.categoryId;

        const db = await getConnection();

        const result = await db.request().input("categoryId", categoryId)
            .query(`
                SELECT *    
                FROM Products
                WHERE CategoryId = @categoryId
                ORDER BY Price DESC
            `);

        res.json({
            tool: "get_products",
            categoryId: categoryId,
            count: result.recordset.length,
            products: result.recordset
        });

    } catch (error) {

        console.error(error);

        res.status(500).json({
            error: error.message
        });
  } 
});

app.listen(3000, () => {

    console.log("MCP Server running on port 3000");
});