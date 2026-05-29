import express from "express";
import { getConnection } from "./db.js";

const app = express();

app.use(express.json());

/*
    MCP Tool:
    get_products()
    get_categories()
    get_products_by_category(category_id)
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
            tool: "get_products_by_category",
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

app.get("/categories/:categoryId/:topN", async (req, res) => {
    try {
        const { categoryId, topN } = req.query;

        const db = await getConnection();

        const result = await db.request()
            .input("categoryId", categoryId)
            .input("topN", topN)
            .query(`
                SELECT TOP (@topN) *
                FROM Products
                WHERE CategoryId = @categoryId
                ORDER BY Price DESC
            `);

        res.json({
            tool: "get_top_products",
            categoryId: categoryId,
            topN: topN,
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

    console.log("Product REST API Server running on port 3000");
});